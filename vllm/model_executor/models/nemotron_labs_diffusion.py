# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronLabsDiffusion model wiring for vLLM (draft scaffolding).

This module onboards ``nvidia/Nemotron-Labs-Diffusion-8B`` (and the public
3B TinyStories sibling) on top of the diffusion runtime introduced in
``[Model] Add DiffusionGemma Support`` (vllm/pull/45163). The two HF
checkpoints share the same architecture name (``NemotronLabsDiffusionModel``)
and the same Ministral-3-based bidirectional masked-diffusion design — only
the transformer width and depth differ.

What this file contains
-----------------------
* ``NemotronLabsDiffusionForBlockDiffusion`` — the registered model class.
  Reuses vLLM's existing ``LlamaForCausalLM`` backbone for the transformer
  body (Ministral-3 is structurally Llama: dense MLP, GQA, RoPE, RMSNorm)
  and adds a ``diffusion_head`` linear over the encoder's final hidden state.
* HF weight-name remap: ``encoder.* → model.*``, fused QKV / gate-up stacking,
  and ``diffusion_head.weight`` passthrough.
* ModelState delegation to ``DiffusionGemmaModelState`` so the same denoising
  loop (canvas, stability, entropy-bound acceptance) drives this model. The
  Nemotron checkpoint does not ship a self-conditioning MLP; that branch is
  skipped at runtime by leaving ``self.self_conditioning = None``.

Known follow-ups (intentional in this draft)
--------------------------------------------
* End-to-end smoke-serving against the 8B / 3B checkpoints — requires the
  ModelState refactor to make ``self_conditioning`` optional rather than
  unconditional (the gemma path always calls it).
* Tests under ``tests/models/registry.py`` and a 3B TinyStories smoke test
  mirroring the SGLang one in this branch's sibling repo.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionSampler,
)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from .interfaces import SupportsPP, SupportsQuant

logger = init_logger(__name__)


class NemotronLabsDiffusionForBlockDiffusion(nn.Module, SupportsQuant, SupportsPP):
    """Nemotron Labs Diffusion model registered as block-diffusion.

    Backbone: ``LlamaModel`` (Ministral-3 is layer-compatible with Llama).
    Head: a single dense ``diffusion_head`` projecting the encoder's last
    hidden state to vocabulary logits. The model never auto-regressively
    predicts; all sampling is mediated by the diffusion ModelState/Sampler.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Nemotron HF checkpoints store the transformer under ``encoder.*``
            # alongside a top-level ``diffusion_head.weight``. vLLM's Llama
            # backbone expects ``model.*``.
            "encoder.": "model.",
        },
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

        # Transformer body. Llama backbone matches Ministral-3 (GQA, RoPE,
        # RMSNorm, dense SwiGLU MLP). vLLM picks attention causality from
        # the per-request metadata supplied by the diffusion ModelState, so
        # the same backbone serves both the bidirectional drafting pass and
        # the causal verify / KV-update pass.
        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.diffusion_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size, soft_cap=None)

        # Nemotron-Labs-Diffusion does not have a self-conditioning MLP.
        # Setting this to None signals the shared ModelState to skip the
        # self-conditioning blend during denoising.
        self.self_conditioning = None

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load Nemotron HF weights into the vLLM Llama backbone.

        Mapping (HF → vLLM):
            encoder.embed_tokens.weight                 → model.embed_tokens.weight
            encoder.layers.N.self_attn.{q,k,v}_proj.*   → model.layers.N.self_attn.qkv_proj.* (stacked)
            encoder.layers.N.self_attn.o_proj.*         → model.layers.N.self_attn.o_proj.*
            encoder.layers.N.mlp.{gate,up}_proj.*       → model.layers.N.mlp.gate_up_proj.* (stacked)
            encoder.layers.N.mlp.down_proj.*            → model.layers.N.mlp.down_proj.*
            encoder.layers.N.{input,post_attention}_layernorm.weight → ... unchanged
            encoder.norm.weight                         → model.norm.weight
            diffusion_head.weight                       → diffusion_head.weight
        """
        from vllm.model_executor.models.llama import LlamaForCausalLM

        # Delegate the encoder.* → model.* prefix mapping and qkv / gate_up
        # stacking to LlamaForCausalLM.load_weights, which already understands
        # ``packed_modules_mapping``. The diffusion_head weight tensor passes
        # through unchanged.
        return LlamaForCausalLM.load_weights(self, weights)


class NemotronLabsDiffusionModelState(DiffusionGemmaModelState):
    """ModelState for Nemotron Labs Diffusion.

    Subclasses ``DiffusionGemmaModelState`` because the canvas / step /
    history bookkeeping is shared across block-diffusion models. The
    Nemotron checkpoint lacks the self-conditioning MLP that gemma uses,
    so any branch that depends on ``self.model.self_conditioning`` falls
    back to identity behavior here (the parent class skips the MLP when
    ``self.model.self_conditioning is None``).
    """

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        # Reuse the gemma DiffusionSampler. It reads ``embed_weight`` from
        # the backbone for soft-embedding lookups — Llama embed_tokens
        # exposes the same attribute, so no Nemotron-specific subclass is
        # required at this stage.
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config or {}
        sampler_cfg = gen.get("sampler_config") or {}
        entropy_bound = sampler_cfg.get("entropy_bound", 0.0) or 0.0
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=float(gen.get("t_min", 0.0)),
            t_max=float(gen.get("t_max", 1.0)),
            entropy_bound=float(entropy_bound),
            confidence_threshold=float(gen.get("confidence_threshold", 0.0)),
            # Nemotron-Labs models keep the embedding weight under
            # ``model.embed_tokens``; the gemma sampler reads
            # ``embed_weight`` for soft-embedding lookups.
            embed_weight=self.model.model.embed_tokens.weight,
            # No backbone-level activation normalizer in Nemotron-Labs.
            normalizer=torch.tensor(1.0, device=self.device),
        ), None
