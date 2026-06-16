# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Labs Diffusion model for vLLM.

Onboards ``nvidia/Nemotron-Labs-Diffusion-8B`` (and the public 3B
TinyStories sibling) on top of the block-diffusion runtime introduced in
#45163. The transformer body is Ministral-3-based (Llama-style GQA, YARN
RoPE, RMSNorm, SwiGLU MLP) with one Nemotron-specific addition — the
Llama-4 per-token Q scaling applied post-RoPE. ``ar_mode=true`` on the
HF config switches attention from bidirectional (block-diffusion default)
to fully causal so a plain AR decoder path matches the SGLang reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionSampler,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsPP, SupportsQuant
from .utils import (
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _llama4_q_scale(
    positions: torch.Tensor, beta: float, max_pos: int
) -> torch.Tensor:
    """Per-token Q scale = 1 + beta * log(1 + floor(pos / max_pos))."""
    return (
        1.0 + beta * torch.log(1.0 + torch.floor(positions.float() / max_pos))
    ).unsqueeze(-1)


class NemotronLabsDiffusionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NemotronLabsDiffusionAttention(nn.Module):
    """Llama-style GQA attention + Llama-4 per-position Q scaling.

    ``causal=True`` (the ``ar_mode`` toggle on the HF config) selects a
    standard causal Attention block; otherwise an EncoderOnlyAttention
    runs the bidirectional block-diffusion encoder pass.
    """

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        causal: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        rope_params = getattr(config, "rope_parameters", None) or {}
        self.llama4_beta: float | None = rope_params.get("llama_4_scaling_beta")
        self.max_pos = int(
            rope_params.get(
                "original_max_position_embeddings",
                getattr(config, "max_position_embeddings", 16384),
            )
        )

        attention_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 16384),
            rope_parameters=rope_params or None,
            is_neox_style=True,
        )

        attn_type = AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
        attn_cls = Attention if causal else EncoderOnlyAttention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if self.llama4_beta is not None:
            scale = _llama4_q_scale(positions, self.llama4_beta, self.max_pos).to(
                q.dtype
            )
            q = q.view(-1, self.num_heads, self.head_dim)
            q = (q * scale.unsqueeze(1)).view(-1, self.num_heads * self.head_dim)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronLabsDiffusionDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        causal = bool(getattr(config, "ar_mode", False))

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NemotronLabsDiffusionAttention(
            config,
            quant_config,
            prefix=f"{prefix}.self_attn",
            causal=causal,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = NemotronLabsDiffusionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class NemotronLabsDiffusionTransformer(nn.Module):
    """Ministral-3 transformer body with optional ar_mode causal switch."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NemotronLabsDiffusionDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class NemotronLabsDiffusionForBlockDiffusion(nn.Module, SupportsQuant, SupportsPP):
    """Nemotron Labs Diffusion served via the vLLM block-diffusion runtime.

    Backbone: ``NemotronLabsDiffusionTransformer`` (Ministral-3 with the
    Llama-4 Q scaling). Head: ``diffusion_head`` linear over the encoder's
    last hidden state. The HF checkpoint stores the transformer under
    ``encoder.*`` alongside a top-level ``diffusion_head.weight``.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model."},
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        return NemotronLabsDiffusionModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_dtype = vllm_config.model_config.dtype

        self.model = NemotronLabsDiffusionTransformer(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.diffusion_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size, soft_cap=None)

        # Nemotron has no self-conditioning MLP. The shared diffusion
        # ModelState (see diffusion_gemma._apply_self_conditioning) skips
        # the SC mixing step when this attribute is None.
        self.self_conditioning = None

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self, input_ids: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.diffusion_head, hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Apply HF→vLLM name remap and Llama-style stacked-param loading.

        Stacks separate ``q/k/v_proj`` weights into ``qkv_proj`` and
        ``gate/up_proj`` into ``gate_up_proj``, then loads the rest by name.
        Matches the loader used by ``LlamaForCausalLM``.
        """
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        mapped = self.hf_to_vllm_mapper.apply(weights)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in mapped:
            if "rotary_emb.inv_freq" in name:
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                param = params_dict[stacked_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning("Skipping unknown weight: %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class NemotronLabsDiffusionModelState(DiffusionGemmaModelState):
    """ModelState for Nemotron Labs Diffusion.

    Subclasses the shared DiffusionGemmaModelState. The Nemotron checkpoint
    lacks a self-conditioning MLP and an embedding normalizer; the parent
    class skips the SC mixing step when ``self.model.self_conditioning is
    None`` (see diffusion_gemma._apply_self_conditioning).
    """

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config or {}
        sampler_cfg = gen.get("sampler_config") or {}
        entropy_bound = float(sampler_cfg.get("entropy_bound", 0.0) or 0.0)
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=float(gen.get("t_min", 0.0)),
            t_max=float(gen.get("t_max", 1.0)),
            entropy_bound=entropy_bound,
            confidence_threshold=float(gen.get("confidence_threshold", 0.0)),
            embed_weight=self.model.model.embed_tokens.weight,
            normalizer=torch.tensor(1.0, device=self.device),
        ), None
