#!/usr/bin/env python3
"""Focused regression tests for the official segmentation protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from eval.eval_official_segmentation import (
    IGNORE_LABEL,
    SegmentationRecord,
    checkpoint_payload,
    learning_rate_at_step,
    load_rgb_mask,
    restore_checkpoint,
    resize_keep_ratio,
    slide_starts,
)


class SegmentationProtocolTest(unittest.TestCase):
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


    def test_legacy_checkpoint_is_read_only_until_complete(self) -> None:
        args = SimpleNamespace(
            dataset="voc",
            model_id="model",
            processor_id=None,
            steps=10,
            evaluate_only=False,
        )
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(head.parameters())
        payload = checkpoint_payload(args, 5, head, optimizer, -1.0, [])
        self.assertEqual(payload["resume_protocol"], "absolute-step-v1")
        payload.pop("resume_protocol")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Cannot deterministically continue"):
                restore_checkpoint(path, args, head, optimizer)

            args.evaluate_only = True
            step, _, _ = restore_checkpoint(path, args, head, optimizer)
            self.assertEqual(step, 5)

            args.evaluate_only = False
            args.steps = 5
            step, _, _ = restore_checkpoint(path, args, head, optimizer)
            self.assertEqual(step, 5)


if __name__ == "__main__":
    unittest.main()
