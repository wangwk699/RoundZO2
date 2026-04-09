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
from typing import Callable, Optional, Union
from transformers.cache_utils import Cache, DynamicCache

class QuantLlama3MLP(nn.Module):
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


class QuantLlama3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = QuantLinear(org_module.q_proj, args.weight_quant_params, args.act_quant_params)
        self.k_proj = QuantLinear(org_module.k_proj, args.weight_quant_params, args.act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj, args.weight_quant_params, args.act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj, args.weight_quant_params, args.act_quant_params)


        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)

        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # QK^T MatMul (量化)
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))


        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout=0.0 if not self.training else self.attention_dropout
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)        
        
        # PV MatMul (量化)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()            
        
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantLlama3DecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args)
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
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
            smooth_ln_fcs_temporary(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.o_proj,
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
                module.use_temporary_parameter=True

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
            smooth_ln_fcs_inplace(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

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