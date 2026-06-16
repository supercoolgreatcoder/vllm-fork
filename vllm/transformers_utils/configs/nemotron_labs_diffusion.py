# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class NemotronLabsDiffusionConfig(PretrainedConfig):
    """HF config for `nvidia/Nemotron-Labs-Diffusion-8B` and the public 3B
    TinyStories sibling.

    Ministral-3-based bidirectional masked-diffusion LM with a plain
    ``diffusion_head`` linear over the encoder's last hidden state. YARN-scaled
    rope at a 262k max-position budget.
    """

    model_type = "nemotron_labs_diffusion"

    block_size: int = 32
    mask_id: int = 100

    def __init__(self, **kwargs: Any) -> None:
        # Extract our extras early so super().__init__ does not see unknown
        # attrs as warnings.
        block_size = kwargs.pop("block_size", self.block_size)
        mask_id = kwargs.pop("mask_id", self.mask_id)

        # Set max_position_embeddings *before* rope-parameter processing —
        # transformers>=5 calls ``standardize_rope_params`` while parsing
        # ``rope_parameters``, which reads ``self.max_position_embeddings``.
        if "max_position_embeddings" in kwargs:
            self.max_position_embeddings = kwargs["max_position_embeddings"]

        rope_params = kwargs.get("rope_parameters")
        rope_theta_top = kwargs.get("rope_theta")
        if isinstance(rope_params, dict) and rope_theta_top is None:
            rope_theta_top = rope_params.get("rope_theta")
        if isinstance(rope_params, dict) and kwargs.get("rope_scaling") is None:
            kwargs["rope_scaling"] = rope_params

        super().__init__(**kwargs)

        # PretrainedConfig has ``rope_theta = None`` as a class default that
        # the constructor sets when no rope_theta kwarg is passed; we lift it
        # from rope_parameters explicitly here for the vLLM rope path that
        # reads the top-level attribute.
        if rope_theta_top is not None:
            self.rope_theta = rope_theta_top

        self.block_size = block_size
        self.mask_id = mask_id
        # ``canvas_length`` is the canonical field used by vLLM's diffusion
        # runtime (see ModelConfig.is_diffusion); Nemotron's HF config calls
        # it ``block_size``. Mirror the value so vLLM auto-detects this as a
        # diffusion model and routes through the V2 model runner.
        self.canvas_length = block_size
