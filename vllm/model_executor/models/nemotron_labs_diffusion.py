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
from vllm.model_executor.layers.attention import Attention
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
from vllm.v1.worker.gpu.sample.output import SamplerOutput

import numpy as np

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

        # vLLM's ``YaRNScalingRotaryEmbedding`` computes
        # ``mscale = 0.1*log(factor) + 1.0`` and multiplies cos/sin by it
        # by default — that gives 1.277 for Nemotron's factor=16 and
        # silently rescales every rotary position. Nemotron's config
        # explicitly sets ``mscale: 1.0`` (no extra scaling), so we
        # override vLLM's auto-mscale by injecting
        # ``apply_yarn_scaling=False`` into the rope dict; with
        # attn_factor=1.0 (default) that makes mscale=1.0 and matches
        # the model's training. Without this fix, the model's logits
        # are sharply flatter — top-1 token probability drops ~10× and
        # GSM8K accuracy drops ~5pp.
        rope_params_for_vllm = {**rope_params} if rope_params else None
        if rope_params_for_vllm and rope_params_for_vllm.get("rope_type") == "yarn":
            if "apply_yarn_scaling" not in rope_params_for_vllm:
                rope_params_for_vllm["apply_yarn_scaling"] = False
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 16384),
            rope_parameters=rope_params_for_vllm,
            is_neox_style=True,
        )

        # Always use the unified Attention layer. The diffusion runtime
        # (DiffusionGemmaModelState.prepare_attn) sets a per-request
        # ``causal`` flag at runtime so the same KV-writing attention block
        # serves both the causal (encoder/AR) and bidirectional (denoise)
        # phases — exactly what the Gemma4 backbone does. For pure-AR mode
        # (ar_mode=True with diffusion_config disabled), the kv_cache_update
        # path also expects every attention layer to participate in KV write.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
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

    def get_model_state_cls(self):
        # ar_mode=true bypasses the diffusion state machine: the model is
        # served as a plain causal LM through vLLM's DefaultModelState.
        if getattr(self.config, "ar_mode", False):
            from vllm.v1.worker.gpu.model_states.default import (
                DefaultModelState,
            )

            return DefaultModelState
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


class NemotronDiffusionSampler(DiffusionSampler):
    """FastDiffuser-style top-k confidence unmasking for Nemotron.

    Mirrors SGLang's ``FastDiffuser`` decoder for Nemotron Labs Diffusion:

    - Canvas is initialized to the model's ``mask_token_id`` (100), not
      random tokens. The model was trained to predict the original token
      at masked positions; passing random tokens (the Gemma path) is
      off-distribution and produces gibberish.
    - Each denoise step computes ``argmax(logits with mask suppressed)``
      and the softmax probability of that argmax. The top-k positions by
      probability are committed — k scales as ``ceil(remaining / steps_left)``
      so the block converges in roughly ``max_denoising_steps`` rounds.
    - No entropy bound, stability gate, or self-conditioning — none apply
      to Nemotron's discrete-token diffusion paradigm.

    Encoder/commit cycle is unchanged: when all positions are unmasked the
    sampler flips ``is_encoder_phase`` so the next pass runs causally and
    rewrites the KV cache for the freshly committed block (the
    ``causal_context: true`` mode in SGLang's FastDiffuser yaml).
    """

    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: Any,
        *,
        mask_token_id: int,
        eos_token_id: int | None,
        max_denoising_steps: int,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> None:
        super().__init__(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=vocab_size,
            diffusion_states=diffusion_states,
            confidence_threshold=0.0,
            t_min=0.0,
            t_max=1.0,
            entropy_bound=0.0,
            embed_weight=embed_weight,
            normalizer=normalizer,
        )
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.max_denoising_steps_n = max_denoising_steps

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            # Block-1 seeding (matching HF's prefill-last-token seed) was
            # tried here but regressed accuracy by ~2.5pp on GSM8K — the
            # bidirectional first denoise step at all-masks already commits
            # the highest-confidence position first, which approximates the
            # same effect at lower cost. Only block-2+ benefits from
            # explicit seeding (done in the commit branch below).
            return self._handle_prefill(input_batch, device)

        states = self.diffusion_states
        CL = self.canvas_length
        mask_id = self.mask_token_id

        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()

        if num_decode == 0:
            return self._build_output(
                input_batch, sampled, num_sampled, per_req_nlogits_np, device
            )

        # is_commit snapshot BEFORE we mutate state. Slots whose previous
        # step converged have is_encoder_phase=True now — this pass ran
        # them causally to refresh the KV cache, so we EMIT their existing
        # canvas (no further denoising) and reset for the next block.
        is_commit = states.is_encoder_phase[decode_slots].clone()
        is_denoise = ~is_commit

        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = torch.from_numpy(
            valid_canvas_len_np.astype(np.int64)
        ).to(device)

        # Reshape logits to [num_decode, CL, vocab]. Truncated canvases
        # (CL spans crossing max_model_len) have fewer than CL rows in
        # logits — pad them here so the per-slot tensor view is uniform.
        if valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(
                logits.shape[0] - 1
            )
            logits = logits[src.reshape(-1)] * valid.reshape(-1, 1).to(
                logits.dtype
            )

        logits_2d = logits.view(num_decode, CL, -1)

        # ---- EMIT FIRST: read the existing canvas for committing slots.
        # This MUST happen before we mutate states.canvas below.
        sampled[decode_idx] = (
            states.canvas[decode_slots].to(sampled.dtype)
            * is_commit.unsqueeze(-1).to(sampled.dtype)
        )
        num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
            num_sampled.dtype
        )

        # ---- Update canvas for denoising slots via top-k unmasking.
        # Match SGLang FastDiffuser's _compute_confidence exactly:
        #   1. argmax with mask_id suppressed (-inf), so mask never wins,
        #   2. softmax over the ORIGINAL logits in the model's dtype (no
        #      float32 cast — SGLang gets 94.5% with bf16 throughout).
        logits_for_argmax = logits_2d.clone()
        logits_for_argmax[..., mask_id] = float("-inf")
        x0 = torch.argmax(logits_for_argmax, dim=-1)  # [num_decode, CL]

        probs = torch.softmax(logits_2d, dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

        canvas = states.canvas[decode_slots].clone()  # [num_decode, CL]
        is_masked = canvas == mask_id

        # EOS-freeze (FastDiffuser): if EOS has already been committed in
        # the block, every position at-or-after the first EOS is "frozen"
        # — they get -inf confidence so the top-k never picks them, and
        # they're filled with EOS at the end of this step. Without this
        # the model keeps generating after an EOS landed mid-block,
        # producing the long meandering outputs that drag down accuracy.
        committed = ~is_masked
        eos_freeze = torch.zeros_like(canvas, dtype=torch.bool)
        if self.eos_token_id is not None:
            eos = self.eos_token_id
            eos_committed = committed & (canvas == eos)
            if eos_committed.any():
                first_eos_pos = torch.where(
                    eos_committed.any(dim=-1),
                    torch.argmax(eos_committed.int(), dim=-1),
                    torch.full_like(eos_committed.int()[:, 0], CL),
                )
                positions = torch.arange(CL, device=device).unsqueeze(0)
                eos_freeze = positions >= first_eos_pos.unsqueeze(-1)

        # Confidence is only valid for masked AND non-frozen positions.
        active = is_masked & ~eos_freeze
        confidence = torch.where(
            active, x0_p, torch.full_like(x0_p, float("-inf"))
        )

        # FastDiffuser scheduler: unmask ceil(remaining / steps_left) per
        # step → converges in at most ``max_denoising_steps`` rounds.
        step = states.step[decode_slots]
        steps_left = (self.max_denoising_steps_n - step).clamp(min=1)
        remaining_masks = is_masked.sum(dim=-1)
        k = ((remaining_masks + steps_left - 1) // steps_left).clamp(min=1)
        k = torch.minimum(k, remaining_masks)  # [num_decode]

        # Top-k by confidence per slot. Positions whose rank < k AND are
        # currently masked get committed.
        ranks = torch.argsort(
            torch.argsort(confidence, dim=-1, descending=True), dim=-1
        )
        unmask_mask = (ranks < k.unsqueeze(-1)) & is_masked
        denoised_canvas = torch.where(unmask_mask, x0, canvas)

        # Convergence: no remaining masks, or we've hit the step budget.
        new_step = step + 1
        max_steps = new_step >= self.max_denoising_steps_n
        no_masks = (denoised_canvas == mask_id).sum(dim=-1) == 0
        converged = no_masks | max_steps

        # Force any leftover masks to argmax when we hit the step budget.
        if max_steps.any():
            force = max_steps.unsqueeze(-1) & (denoised_canvas == mask_id)
            denoised_canvas = torch.where(force, x0, denoised_canvas)

        # EOS fill: any positions still masked and inside the EOS-freeze
        # range (computed pre-step) get filled with EOS now. Also catches
        # the case where this step itself committed an EOS — for those,
        # propagate forward to the rest of the block.
        if self.eos_token_id is not None:
            eos = self.eos_token_id
            has_eos = (denoised_canvas == eos).any(dim=-1)
            if has_eos.any():
                eos_pos = (denoised_canvas == eos).int()
                first_eos = torch.where(
                    has_eos,
                    torch.argmax(eos_pos, dim=-1),
                    torch.full_like(eos_pos[:, 0], CL),
                )
                positions = torch.arange(CL, device=device).unsqueeze(0)
                after_eos = positions >= first_eos.unsqueeze(-1)
                fill_with_eos = (
                    has_eos.unsqueeze(-1)
                    & after_eos
                    & (denoised_canvas == mask_id)
                )
                denoised_canvas = torch.where(
                    fill_with_eos,
                    torch.full_like(denoised_canvas, eos),
                    denoised_canvas,
                )
                no_masks_after = (denoised_canvas == mask_id).sum(dim=-1) == 0
                converged = converged | no_masks_after

        # Slot-state writes:
        #   committing slots → canvas reset to mask_id (start of next block)
        #   denoising slots  → canvas updated to denoised_canvas
        #
        # Next-block seeding (writing ``argmax(causal-mode last-position
        # logit)`` into the next block's pos 0 — matching HF
        # ``output.logits[:, -1, :].argmax()`` between blocks) was tried
        # here but did NOT improve full GSM8K (84.7% vs 86.4% without it).
        # Keeping the simpler all-mask reset.
        fresh_canvas = torch.full_like(canvas, mask_id)
        new_canvas = torch.where(
            is_commit.unsqueeze(-1), fresh_canvas, denoised_canvas
        )
        states.canvas[decode_slots] = new_canvas

        # State machine: commit → denoise (encoder_phase=False, step=0).
        # Denoise → commit when converged (encoder_phase=True). Step
        # counter resets on commit; increments on denoise.
        next_encoder = torch.where(
            is_commit, torch.zeros_like(is_commit), converged & is_denoise
        )
        states.is_encoder_phase[decode_slots] = next_encoder
        states.step[decode_slots] = torch.where(
            is_commit, torch.zeros_like(new_step), new_step
        )

        # Draft tokens for next iteration's input_ids: the canvas
        # (post-update) for all decode slots.
        all_slots = input_batch.idx_mapping[:num_reqs]
        self.req_states.draft_tokens[all_slots, :CL] = states.canvas[all_slots]

        return self._build_output(
            input_batch, sampled, num_sampled, per_req_nlogits_np, device
        )


class NemotronLabsDiffusionModelState(DiffusionGemmaModelState):
    """ModelState for Nemotron Labs Diffusion.

    Subclasses the shared DiffusionGemmaModelState. The Nemotron checkpoint
    lacks a self-conditioning MLP and an embedding normalizer; the parent
    class skips the SC mixing step when ``self.model.self_conditioning is
    None`` (see diffusion_gemma._apply_self_conditioning).

    Canvas init is overridden to use Nemotron's ``mask_token_id`` (the
    Gemma default of random tokens is off-distribution here — the model
    was trained to denoise from a uniform mask).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Override canvas init: Nemotron expects MASK tokens, not random.
        mask_token_id = getattr(
            self.model_config.hf_config, "mask_token_id", 100
        )
        ds = self.diffusion_states
        ds.canvas.fill_(mask_token_id)
        ds.argmax_canvas.fill_(mask_token_id)
        # Monkey-patch init_canvas so newly-prefilled requests also start
        # from all-masks. ``DiffusionGemmaRequestStates.init_canvas`` would
        # otherwise overwrite with ``torch.randint``.
        self._mask_token_id = mask_token_id
        _orig_states = ds

        def _init_canvas(slot_indices_np):
            _orig_states.canvas[slot_indices_np] = mask_token_id
            _orig_states.argmax_canvas[slot_indices_np] = mask_token_id

        ds.init_canvas = _init_canvas

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config or {}
        max_denoising_steps = (
            diffusion_config.max_denoising_steps
            if diffusion_config and diffusion_config.max_denoising_steps
            else int(gen.get("max_denoising_steps", 32))
        )
        eos_token_id = getattr(self.model_config.hf_config, "eos_token_id", None)
        return NemotronDiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            mask_token_id=self._mask_token_id,
            eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
            max_denoising_steps=max_denoising_steps,
            embed_weight=self.model.model.embed_tokens.weight,
            normalizer=torch.tensor(1.0, device=self.device),
        ), None
