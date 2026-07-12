#!/usr/bin/env python3
"""Regression tests for the resumable grounding feature cache."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from eval.eval_official_grounding import create_cached_loader, evaluate_split
from eval.official_grounding_cache import (
    CachedGroundingDataset,
    CompositeGroundingFeatureCache,
    build_feature_cache,
)
from eval.official_grounding_data import (
    IMAGE_TRANSFORM_NAME,
    image_to_tensor,
    load_grounding_records,
)
from eval.official_probe_utils import FrozenVisionTower, VisionMetadata


class GroundingFeatureCacheTest(unittest.TestCase):
    def test_image_resize_matches_big_vision_uint8_contract(self) -> None:
        pixels = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
        image = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2))
        actual = (image_to_tensor(image, 3) * 255).to(torch.uint8)
        expected = torch.tensor(
            [[0, 32, 64], [64, 111, 159], [128, 191, 255]],
            dtype=torch.uint8,
        )
        self.assertTrue(torch.equal(actual[0], expected))
        self.assertTrue(torch.equal(actual[1], expected))
        self.assertTrue(torch.equal(actual[2], expected))

    def _tower(self, family: str = "siglip") -> FrozenVisionTower:
        class DummyVision(torch.nn.Module):
            def __init__(self, model_family: str) -> None:
                super().__init__()
                self.calls = 0
                self.model_family = model_family
                self.post_layernorm = torch.nn.Identity()

            def forward(self, pixel_values: torch.Tensor, **kwargs):
                del kwargs
                self.calls += 1
                batch, _, height, width = pixel_values.shape
                base = pixel_values.mean(dim=(1, 2, 3)).view(batch, 1, 1)
                token_count = height * width + int(self.model_family == "clip")
                tokens = torch.arange(token_count * 2, device=pixel_values.device)
                tokens = tokens.view(1, token_count, 2).float() + base
                return SimpleNamespace(last_hidden_state=tokens)

        tower = FrozenVisionTower.__new__(FrozenVisionTower)
        torch.nn.Module.__init__(tower)
        tower.device_name = "cpu"
        tower.metadata = VisionMetadata(
            model_id="dummy",
            processor_id="dummy",
            family=family,
            hidden_size=2,
            patch_size=1,
            num_hidden_layers=1,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
        tower.vision_model = DummyVision(family)
        tower.register_buffer("image_mean", torch.zeros(1, 3, 1, 1), persistent=False)
        tower.register_buffer("image_std", torch.ones(1, 3, 1, 1), persistent=False)
        return tower

    def test_clip_cache_retains_pre_pooling_cls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(
                image_root / "image.png"
            )
            records_path = root / "records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "dataset": "refcoco",
                        "split": "train",
                        "split_by": "unc",
                        "ref_id": 1,
                        "sentence_id": 1,
                        "ann_id": 1,
                        "image_id": 1,
                        "file_name": "image.png",
                        "width": 2,
                        "height": 2,
                        "bbox_xywh": [0, 0, 1, 1],
                        "expression": "object",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cache = build_feature_cache(
                root / "cache",
                self._tower("clip"),
                records_path,
                image_root,
                resolution=2,
                batch_size=1,
                workers=0,
                autocast_factory=nullcontext,
                flush_interval=1,
            )
            self.assertEqual(cache.metadata["version"], 5)
            self.assertIn("plus leading CLS", cache.metadata["token_contract"])
            self.assertEqual(cache.shape, (1, 5, 2))

    def test_build_read_and_reuse_preserves_bfloat16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            records_path = root / "records.jsonl"
            rows = []
            for index, value in enumerate((64, 192), start=1):
                file_name = f"image_{index}.png"
                pixels = np.full((2, 2, 3), value, dtype=np.uint8)
                Image.fromarray(pixels).save(image_root / file_name)
                rows.append(
                    {
                        "dataset": "refcoco",
                        "split": "train",
                        "split_by": "unc",
                        "ref_id": index,
                        "sentence_id": index,
                        "ann_id": index,
                        "image_id": index,
                        "file_name": file_name,
                        "width": 2,
                        "height": 2,
                        "bbox_xywh": [0, 0, 1, 1],
                        "expression": f"object {index}",
                    }
                )
            records_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            tower = self._tower()
            cache_dir = root / "cache"
            cache = build_feature_cache(
                cache_dir,
                tower,
                records_path,
                image_root,
                resolution=2,
                batch_size=1,
                workers=0,
                autocast_factory=nullcontext,
                flush_interval=1,
            )
            records = load_grounding_records(records_path)
            features = cache.get(records, "cpu")
            self.assertEqual(cache.metadata["version"], 4)
            self.assertEqual(cache.metadata["image_transform"], IMAGE_TRANSFORM_NAME)
            self.assertIn("no input padding", cache.metadata["token_contract"])
            self.assertIn("CLS/MAP excluded", cache.metadata["token_contract"])
            self.assertEqual(features.dtype, torch.bfloat16)
            self.assertEqual(features.shape, (2, 4, 2))
            self.assertNotEqual(float(features[0, 0, 0]), float(features[1, 0, 0]))
            self.assertEqual(tower.vision_model.calls, 2)

            reused = build_feature_cache(
                cache_dir,
                tower,
                records_path,
                image_root,
                resolution=2,
                batch_size=1,
                workers=0,
                autocast_factory=nullcontext,
                flush_interval=1,
            )
            self.assertEqual(tower.vision_model.calls, 2)
            self.assertTrue(torch.equal(reused.get(records, "cpu"), features))

            class DummyTokenizer:
                max_length = 64
                pad_id = 0
                eos_id = 1

                def encode(self, text: str, add_eos: bool = False) -> list[int]:
                    del self, text, add_eos
                    return [2, 3]


                def decode(self, token_ids: list[int]) -> str:
                    del self, token_ids
                    return "invalid"

            class DummyDecoder:
                seen_shape: tuple[int, ...] | None = None

                def eval(self):
                    return self

                def generate(
                    self,
                    vision_tokens: torch.Tensor,
                    prompts: list[list[int]],
                    pad_id: int,
                    eos_id: int,
                    max_new_tokens: int,
                ) -> list[list[int]]:
                    del pad_id, max_new_tokens
                    self.seen_shape = tuple(vision_tokens.shape)
                    return [list(prompt) + [eos_id] for prompt in prompts]

            cached_dataset = CachedGroundingDataset(records_path)
            cached_loader = create_cached_loader(
                cached_dataset,
                batch_size=2,
                workers=0,
                shuffle=False,
                drop_last=False,
            )
            decoder = DummyDecoder()
            calls_before_eval = tower.vision_model.calls
            metrics = evaluate_split(
                tower,
                decoder,
                DummyTokenizer(),
                cached_loader,
                device="cpu",
                dtype_name="fp32",
                max_new_tokens=1,
                feature_cache=cache,
            )
            self.assertEqual(metrics["expressions"], 2)
            self.assertEqual(metrics["invalid_predictions"], 2)
            self.assertEqual(decoder.seen_shape, (2, 4, 2))
            self.assertEqual(tower.vision_model.calls, calls_before_eval)


    def test_composite_cache_routes_disjoint_images_in_batch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            rows = []
            for image_id, value in ((11, 32), (22, 224)):
                file_name = f"image_{image_id}.png"
                Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(
                    image_root / file_name
                )
                rows.append(
                    {
                        "dataset": "refcoco",
                        "split": "train",
                        "split_by": "unc",
                        "ref_id": image_id,
                        "sentence_id": image_id,
                        "ann_id": image_id,
                        "image_id": image_id,
                        "file_name": file_name,
                        "width": 2,
                        "height": 2,
                        "bbox_xywh": [0, 0, 1, 1],
                        "expression": f"object {image_id}",
                    }
                )

            caches = []
            record_groups = []
            tower = self._tower()
            for index, row in enumerate(rows):
                records_path = root / f"records_{index}.jsonl"
                records_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                caches.append(
                    build_feature_cache(
                        root / f"cache_{index}",
                        tower,
                        records_path,
                        image_root,
                        resolution=2,
                        batch_size=1,
                        workers=0,
                        autocast_factory=nullcontext,
                        flush_interval=1,
                    )
                )
                record_groups.append(load_grounding_records(records_path))

            composite = CompositeGroundingFeatureCache(caches)
            reversed_records = [record_groups[1][0], record_groups[0][0]]
            composite.validate_coverage(reversed_records)
            actual = composite.get(reversed_records, "cpu")
            expected = torch.cat(
                (
                    caches[1].get(record_groups[1], "cpu"),
                    caches[0].get(record_groups[0], "cpu"),
                ),
                dim=0,
            )
            self.assertTrue(torch.equal(actual, expected))
            with self.assertRaisesRegex(RuntimeError, "multiple composite"):
                CompositeGroundingFeatureCache([caches[0], caches[0]])

            caches[1].metadata["image_transform"] = "incompatible_transform"
            with self.assertRaisesRegex(RuntimeError, "image_transform"):
                CompositeGroundingFeatureCache(caches)


if __name__ == "__main__":
    unittest.main()
