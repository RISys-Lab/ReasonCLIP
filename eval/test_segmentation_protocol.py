#!/usr/bin/env python3
"""Focused regression tests for the segmentation protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from eval.eval_segmentation import (
    DEFAULT_SEED,
    DEFAULT_TORCH_DTYPE,
    IGNORE_LABEL,
    PAPER_REFERENCES,
    RESUME_PROTOCOL,
    BNLinearSegmentationHead,
    SegmentationRecord,
    checkpoint_payload,
    learning_rate_at_step,
    load_rgb_mask,
    restore_checkpoint,
    resize_keep_ratio,
    slide_starts,
)
from eval.probe_utils import FrozenVisionTower, _model_family


class SegmentationProtocolTest(unittest.TestCase):
    def test_canonical_defaults(self) -> None:
        self.assertEqual(DEFAULT_TORCH_DTYPE, "bf16")
        self.assertEqual(DEFAULT_SEED, 42)

    def test_clip_and_siglip_head_dimensions(self) -> None:
        clip_head = BNLinearSegmentationHead(2048, 21)
        siglip1_head = BNLinearSegmentationHead(2304, 150)
        self.assertEqual(clip_head.bn.num_features, 2048)
        self.assertEqual(clip_head.classifier.weight.shape, (21, 2048, 1, 1))
        self.assertEqual(siglip1_head.bn.num_features, 2304)
        self.assertEqual(siglip1_head.classifier.weight.shape, (150, 2304, 1, 1))

    def test_model_family_depends_only_on_architecture(self) -> None:
        clip = SimpleNamespace(config=SimpleNamespace(model_type="clip"), name_or_path="reason-clip")
        siglip = SimpleNamespace(config=SimpleNamespace(model_type="siglip"), name_or_path="")
        native_siglip2 = SimpleNamespace(config=SimpleNamespace(model_type="siglip2"), name_or_path="")

        self.assertEqual(_model_family(clip), "clip")
        self.assertEqual(_model_family(siglip), "siglip")
        self.assertEqual(_model_family(native_siglip2), "siglip")

    def test_dense_token_shape_is_strict_for_each_architecture(self) -> None:
        clip = SimpleNamespace(metadata=SimpleNamespace(family="clip"))
        siglip = SimpleNamespace(metadata=SimpleNamespace(family="siglip"))
        clip_tokens = torch.zeros(1, 5, 3)
        siglip_tokens = torch.zeros(1, 4, 3)

        self.assertEqual(
            FrozenVisionTower._split_patches(clip, clip_tokens, 2, 2).shape,
            (1, 3, 2, 2),
        )
        self.assertEqual(
            FrozenVisionTower._split_patches(siglip, siglip_tokens, 2, 2).shape,
            (1, 3, 2, 2),
        )
        with self.assertRaisesRegex(RuntimeError, "Expected CLS plus"):
            FrozenVisionTower._split_patches(clip, siglip_tokens, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "Expected 4 patch tokens"):
            FrozenVisionTower._split_patches(siglip, clip_tokens, 2, 2)

    def test_paper_references_cover_clip_and_both_siglip_generations(self) -> None:
        self.assertEqual(PAPER_REFERENCES[("openai/clip-vit-large-patch14", "voc")][0], 74.5)
        self.assertEqual(
            PAPER_REFERENCES[("google/siglip-so400m-patch14-384", "ade20k")][0],
            40.8,
        )
        self.assertEqual(
            PAPER_REFERENCES[("google/siglip2-so400m-patch14-384", "ade20k")][0],
            45.4,
        )

    def test_reduce_zero_label_matches_mmseg_ignore_handling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
            Image.fromarray(np.array([[0, 1], [150, 255]], dtype=np.uint8)).save(mask_path)
            record = SegmentationRecord(image_path, mask_path, "sample")

            _, reduced = load_rgb_mask(record, reduce_zero_label=True)

        expected = np.array([[IGNORE_LABEL, 0], [149, IGNORE_LABEL]], dtype=np.uint8)
        np.testing.assert_array_equal(reduced, expected)

    def test_resize_keep_ratio_respects_short_and_long_edges(self) -> None:
        image = np.zeros((100, 1000, 3), dtype=np.uint8)
        resized, mask = resize_keep_ratio(image, None, 512, 2048, 1.0)
        self.assertEqual(resized.shape[:2], (205, 2048))
        self.assertIsNone(mask)

    def test_sliding_windows_cover_last_edge(self) -> None:
        self.assertEqual(slide_starts(512, 512, 341), [0])
        self.assertEqual(slide_starts(1000, 512, 341), [0, 341, 488])

    def test_learning_rate_combines_warmup_and_poly(self) -> None:
        args = SimpleNamespace(
            learning_rate=1e-3,
            steps=40_000,
            warmup_steps=1500,
            warmup_ratio=1e-6,
        )
        self.assertAlmostEqual(learning_rate_at_step(args, 0), 1e-9)
        self.assertAlmostEqual(
            learning_rate_at_step(args, 750),
            1e-3 * (1 - 750 / 40_000) * (1e-6 + (1 - 1e-6) * 0.5),
        )
        self.assertAlmostEqual(
            learning_rate_at_step(args, 1500),
            1e-3 * (1 - 1500 / 40_000),
        )
        self.assertEqual(learning_rate_at_step(args, 40_000), 0.0)

    def test_checkpoint_has_no_processor_override_and_rejects_legacy_format(self) -> None:
        args = SimpleNamespace(
            dataset="voc",
            model_id="model",
        )
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(head.parameters())
        payload = checkpoint_payload(args, 5, head, optimizer, -1.0, [])
        self.assertEqual(payload["resume_protocol"], RESUME_PROTOCOL)
        self.assertNotIn("processor_id", payload)
        payload.pop("resume_protocol")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Incompatible segmentation checkpoint"):
                restore_checkpoint(path, args, head, optimizer)


if __name__ == "__main__":
    unittest.main()
