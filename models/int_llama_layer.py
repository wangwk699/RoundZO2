import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, LlamaRMSNorm, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy
from models.transformation import *


class QuantLlamaMLP(nn.Module):
    def __init__(
            self,
            org_module: nn.Module,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            args=None,
    ):
        super().__init__()
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                     args.weight_quant_params,
                                     args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                     args.weight_quant_params,
                                     args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                   args.weight_quant_params,
                                   args.act_quant_params)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None,
                 layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if layer_idx is None:
            import warnings
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = LlamaRotaryEmbedding(config)

        # FIXME: self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.weight_quant_params, args.act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (self.q_proj(hidden_states)
                        .view(bsz, q_len, self.num_heads, self.head_dim)
                        .transpose(1, 2))
        key_states = (self.k_proj(hidden_states)
                      .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                      .transpose(1, 2))
        value_states = (self.v_proj(hidden_states)
                        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                        .transpose(1, 2))

        # 替换或修正您代码中计算 kv_seq_len 的逻辑为以下内容：
        kv_seq_len = key_states.shape[-2]  # 当前输入序列的长度

        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_length = past_key_values.get_seq_length()
                kv_seq_len += past_length
                # print(
                #     f"[DEBUG] Past length: {past_length}, Current length: {key_states.shape[-2]}, Total KV seq len: {kv_seq_len}")
            else:
                try:
                    if hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > self.layer_idx:
                        past_key_tensor = past_key_values.key_cache[self.layer_idx]
                        if past_key_tensor is not None and hasattr(past_key_tensor, 'shape'):
                            past_length = past_key_tensor.shape[-2]
                            kv_seq_len += past_length
                            # print(f"[DEBUG] Alternative - Past length: {past_length}, Total KV seq len: {kv_seq_len}")
                except Exception as e:
                    # print(f"[DEBUG] Error extracting past length: {e}")
                    raise e

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            # 获取当前层的缓存
            if hasattr(past_key_values, 'get_layer_cache'):
                # 新版API：使用get_layer_cache方法
                layer_cache = past_key_values.get_layer_cache(self.layer_idx)
                if layer_cache is not None:
                    past_key, past_value = layer_cache
                    # 手动拼接
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)
            elif hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                # 旧版API：直接访问key_cache和value_cache
                if (self.layer_idx < len(past_key_values.key_cache) and
                        past_key_values.key_cache[self.layer_idx] is not None):
                    past_key = past_key_values.key_cache[self.layer_idx]
                    past_value = past_key_values.value_cache[self.layer_idx]
                    # 手动拼接
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)

            # 更新缓存
            if use_cache:
                # 使用DynamicCache的标准方法更新
                cache_kwargs = {"sin": sin, "cos": cos} if hasattr(self, 'rotary_emb') else {}
                past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 args,
                 layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
            layer_idx=layer_idx
        )
        self.mlp = QuantLlamaMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = OmniLlamaRMSNorm(ori_layer.input_layernorm,
                                                eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = OmniLlamaRMSNorm(ori_layer.post_attention_layernorm,
                                                         eps=ori_layer.post_attention_layernorm.variance_epsilon)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor]] = None,
            # ldx:add:
            num_items_in_batch=None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # kwargs.pop("num_items_in_batch", None)

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # print('Type of `hidden_states`', type(hidden_states), '-' * 100)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # print(outputs, type(outputs), '-' * 100)

        return outputs[0]

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,
                                    [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                                    self.fc1_smooth_scale, self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj, self.self_attn.o_proj,
                                   self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                 self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,
                                  [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                  self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                                  self.fc1_smooth_scale, self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj, self.self_attn.o_proj,
                                 self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                               self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()