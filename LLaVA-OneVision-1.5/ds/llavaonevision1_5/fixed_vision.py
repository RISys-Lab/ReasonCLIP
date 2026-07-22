"""Fixed-resolution CLIP and SigLIP towers for LLaVA-OneVision-1.5."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)


class FixedVisionProjector(nn.Module):
    """Project patch features into the language model embedding space."""

    def __init__(self, vision_hidden_size: int, text_hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(vision_hidden_size)
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(self.norm(hidden_states))))


class FixedVisionTower(nn.Module):
    """Wrap CLIP/SigLIP behind the vision interface used by this repository."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        if config.vision_tower_type == "clip":
            backbone_config = CLIPVisionConfig.from_dict(config.backbone_config)
            self.vision_model = CLIPVisionModel(backbone_config)
        elif config.vision_tower_type == "siglip":
            backbone_config = SiglipVisionConfig.from_dict(config.backbone_config)
            self.vision_model = SiglipVisionModel(backbone_config)
        else:
            raise ValueError(f"Unsupported fixed vision tower: {config.vision_tower_type}")

        self.merger = FixedVisionProjector(
            vision_hidden_size=config.hidden_size,
            text_hidden_size=config.text_hidden_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vision_model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.vision_model.parameters()).device

    def _select_features(self, outputs) -> torch.Tensor:
        layer = self.config.vision_feature_layer
        if layer == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[layer]

        if self.config.vision_feature_select_strategy == "default":
            if self.config.vision_tower_type != "clip":
                raise ValueError("Only CLIP has a CLS token to drop with the 'default' strategy")
            hidden_states = hidden_states[:, 1:]

        return hidden_states

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        is_verifying: bool = False,
    ) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                "Fixed vision towers expect pixel_values with shape [images, channels, height, width], "
                f"got {tuple(pixel_values.shape)}"
            )

        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=self.config.vision_feature_layer != -1,
            return_dict=True,
        )
        hidden_states = self._select_features(outputs)

        expected_tokens = self.config.image_seq_length
        if hidden_states.shape[1] != expected_tokens:
            raise ValueError(
                f"{self.config.vision_tower_type} produced {hidden_states.shape[1]} patch tokens; "
                f"expected {expected_tokens} for {self.config.image_size}x{self.config.image_size} "
                f"with patch size {self.config.patch_size}"
            )

        if grid_thw is not None:
            expected_grid = (1, self.config.grid_size, self.config.grid_size)
            actual_grids = [tuple(int(value) for value in row) for row in grid_thw]
            if any(grid != expected_grid for grid in actual_grids):
                raise ValueError(f"Fixed vision tower expected image_grid_thw={expected_grid}, got {actual_grids}")

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if is_verifying:
            return hidden_states
        return self.merger(hidden_states)


__all__ = ["FixedVisionProjector", "FixedVisionTower"]
