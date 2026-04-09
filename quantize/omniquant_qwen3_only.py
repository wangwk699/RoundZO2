import torch
import torch.nn as nn
from models.int_qwen_layer import QuantQwenDecoderLayer
from models.int_qwen3_layer import QuantQwen3DecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

from quantize.utils import (
    let_parameters,
    lwc_parameters,
    get_omni_parameters,
    omni_state_dict,
    register_scales_and_zeros,
    smooth_and_quant_temporary,
    smooth_and_quant_inplace,
    clear_temp_variable,
    set_quant_state,
)

# Qwen3 mask helpers
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)


def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = False
    is_qwen = False
    is_qwen2 = False
    is_qwen3 = False

    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1",
        }
        layer_name_prefix = "model.layers"

    elif "qwen" in args.net.lower():
        is_qwen = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

        if "qwen2" in args.net.lower():
            is_qwen2 = True
            DecoderLayer = QuantQwenDecoderLayer
        else:
            is_qwen3 = True
            DecoderLayer = QuantQwen3DecoderLayer

        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1",
        }
        layer_name_prefix = "model.layers"

    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1",
        }
        layer_name_prefix = "model.decoder.layers"

    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"

    elif "mixtral" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"

    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen2/qwen3 now")

    layers[0] = layers[0].to(dev)

    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = lambda: torch.amp.autocast("cuda")

    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0}

    # non-Qwen3 path
    attention_mask = None
    attention_mask_batch = None
    position_ids = None

    # Qwen3 path
    attention_mask_mapping = None
    position_embeddings = None

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_qwen2 = False
            self.is_qwen3 = False

            # Qwen3Model.forward 会先访问 decoder_layer.attention_type
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1

            if self.is_llama:
                if "attention_mask" in kwargs:
                    cache["attention_mask"] = kwargs["attention_mask"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            elif self.is_qwen2:
                if "attention_mask" in kwargs:
                    cache["attention_mask"] = kwargs["attention_mask"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            elif self.is_qwen3:
                if "position_embeddings" in kwargs:
                    cache["position_embeddings"] = kwargs["position_embeddings"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_qwen2 = is_qwen2
    layers[0].is_qwen3 = is_qwen3

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break

            input_ids = batch[0].to(dev)

            try:
                if is_qwen3:
                    # 只构造一次，后续所有层共享同一份 mapping / position_embeddings
                    if attention_mask_mapping is None:
                        inputs_embeds = model.model.embed_tokens(input_ids)

                        past_key_values = None
                        past_seen_tokens = 0
                        cache_position = torch.arange(
                            past_seen_tokens,
                            past_seen_tokens + inputs_embeds.shape[1],
                            device=inputs_embeds.device,
                        )
                        position_ids = cache_position.unsqueeze(0)

                        mask_kwargs = {
                            "config": model.model.config,
                            "input_embeds": inputs_embeds,
                            "attention_mask": None,
                            "cache_position": cache_position,
                            "past_key_values": past_key_values,
                            "position_ids": position_ids,
                        }

                        attention_mask_mapping = {
                            "full_attention": create_causal_mask(**mask_kwargs),
                        }

                        if "sliding_attention" in model.model.config.layer_types:
                            attention_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

                        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

                    # 这里传 mapping，Qwen3Model.forward 会按 decoder_layer.attention_type 取值
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_mapping,
                        position_ids=position_ids,
                    )
                else:
                    model(input_ids)

            except ValueError:
                pass

    # restore first layer
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "falcon" in args.model:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen2/qwen3 now")

    torch.cuda.empty_cache()

    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None

    loss_func = torch.nn.MSELoss()

    if is_llama or is_qwen2:
        attention_mask = cache.get("attention_mask", None)
        position_ids = cache.get("position_ids", None)

        if attention_mask is not None:
            attention_mask_batch = (
                attention_mask.repeat(args.batch_size, 1, 1, 1)
                if args.deactive_amp
                else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
            )
        else:
            logger.info(
                "No attention mask caught from the first layer."
                " Seems that model's attention works without a mask."
            )
            attention_mask_batch = None

    elif is_qwen3:
        position_embeddings = cache.get("position_embeddings", position_embeddings)
        position_ids = cache.get("position_ids", position_ids)

    else:
        position_ids = None

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)

        if "mixtral" in args.net.lower():
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear) and "gate" not in name:
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)

        qlayer = qlayer.to(dev)

        # 选当前层自己的 attention mask
        if is_qwen3:
            layer_attention_mask = attention_mask_mapping[qlayer.attention_type]
        else:
            layer_attention_mask = attention_mask
            layer_attention_mask_batch = attention_mask_batch

        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)

        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    for j in range(args.nsamples):
                        if is_qwen3:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=layer_attention_mask,
                                    position_embeddings=position_embeddings,
                                )
                        elif is_qwen2:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_ids=position_ids,
                            )[0]
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=layer_attention_mask,
                                    position_ids=position_ids,
                                )[0]
                        else:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_ids=position_ids,
                            )[0]
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=layer_attention_mask,
                                    position_ids=position_ids,
                                )[0]

        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = args.let

        use_shift = True
        if (is_llama or is_qwen) or args.abits == 16:
            use_shift = False

        if args.let:
            qlayer.register_parameter(
                "qkt_smooth_scale",
                torch.nn.Parameter(
                    torch.ones(
                        layer.self_attn.q_proj.out_features,
                        device=dev,
                        dtype=dtype,
                    )
                ),
            )

            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)

                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)

                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_shift",
                                torch.nn.Parameter(shift),
                            )
                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_scale",
                                torch.nn.Parameter(scale),
                            )

        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()

            optimizer = torch.optim.AdamW(
                [
                    {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                    {"params": lwc_parameters(qlayer), "lr": args.lwc_lr},
                ],
                weight_decay=args.wd,
            )
            loss_scaler = utils.NativeScalerWithGradNormCount()

            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []

                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size

                    with traincast():
                        if is_qwen3:
                            # 精简版：Qwen3 直接复用 batch=1 的 mask，依赖广播
                            smooth_and_quant_temporary(qlayer, args, True)
                            quant_out = qlayer(
                                quant_inps[index:index + args.batch_size],
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            is_llama_qwen = is_llama or is_qwen
                            smooth_and_quant_temporary(qlayer, args, is_llama_qwen)
                            quant_out = qlayer(
                                quant_inps[index:index + args.batch_size],
                                attention_mask=layer_attention_mask_batch,
                                position_ids=position_ids,
                            )[0]

                        loss = loss_func(fp_inps[index:index + args.batch_size], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index + args.batch_size], quant_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        parameters=get_omni_parameters(qlayer, use_shift),
                    ).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "
                    f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2}"
                )

            clear_temp_variable(qlayer)
            del optimizer

        qlayer.half()

        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)

        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        if is_qwen3:
                            quant_inps[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            quant_inps[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_ids=position_ids,
                            )[0]

            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, "omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

        if args.real_quant:
            assert args.wbits in [2, 3, 4] and args.abits >= 16
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1)
                zeros = zeros.view(dim0, -1)

                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                    )
                else:
                    q_linear = qlinear_triton.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                    )

                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)
                print(f"pack quantized {name} finished")
                del module

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()

    model.config.use_cache = use_cache
    return model