#!/usr/bin/env python3
"""Focused regression tests for the geometry protocol."""

from __future__ import annotations

import io
import inspect
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from eval.eval_navi_depth import (
    GRADIENT_LOSS_WEIGHT as NAVI_DEPTH_GRADIENT_LOSS_WEIGHT,
    LEARNING_RATE as NAVI_DEPTH_LEARNING_RATE,
    SIG_LOSS_WEIGHT as NAVI_DEPTH_SIG_LOSS_WEIGHT,
    WARMUP_STEPS as NAVI_DEPTH_WARMUP_STEPS,
    parse_args as parse_navi_depth_args,
)
from eval.eval_navi_normals import (
    LEARNING_RATE as NAVI_NORMALS_LEARNING_RATE,
    WARMUP_STEPS as NAVI_NORMALS_WARMUP_STEPS,
    build_datasets as build_navi_normals_datasets,
    parse_args as parse_navi_normals_args,
)
from eval.eval_nyuv2_depth import (
    LinearDepthHead,
    parse_args as parse_nyuv2_depth_args,
)
from eval.eval_nyuv2_normals import (
    LEARNING_RATE as NYUV2_NORMALS_LEARNING_RATE,
    WARMUP_STEPS as NYUV2_NORMALS_WARMUP_STEPS,
    parse_args as parse_nyuv2_normals_args,
)
from eval.geometry_common import (
    RunLimits,
    depth_metrics,
    extract_dpt_features,
    extract_final_features,
    gradient_loss,
    learning_rate_at_step,
    normal_loss,
    normal_metrics,
    normalize_and_resize_normals,
    one_cycle_beta1_at_step,
    paper_reference,
    run_limits,
    sig_loss,
)
from eval.geometry_data import (
    NYU_GEONET_BAD_FILES,
    NYU_GEONET_USABLE_SAMPLES,
    NAVIProbeDataset,
    NumpyCompatUnpickler,
    bbox_crop,
    clean_nyu_geonet_instances,
    depth_to_normals,
    normalize_navi_relative_depth,
    nyu_depth_train_transform,
    read_navi_depth,
)
from eval.probe_utils import (
    DeterministicAugmentDataset,
    DeterministicBatchSampler,
    FrozenVisionTower,
    VisionMetadata,
    _cached_snapshot_revision,
    _snapshot_revision_from_path,
)
from eval.tips_dpt import DepthDecoder, NormalsDecoder


class GeometryProtocolTest(unittest.TestCase):
    def test_numpy2_nyu_pickle_modules_load_under_numpy1(self) -> None:
        unpickler = NumpyCompatUnpickler(io.BytesIO())
        resolved = unpickler.find_class("numpy._core.numeric", "_frombuffer")
        self.assertIs(resolved, np.core.numeric._frombuffer)

    def test_navi_dataset_exposes_relative_depth_mode(self) -> None:
        parameters = inspect.signature(NAVIProbeDataset).parameters
        self.assertIn("relative_depth", parameters)
        self.assertTrue(parameters["relative_depth"].default)

    def test_geometry_defaults_use_bf16_and_seed_42(self) -> None:
        entries = (
            ("eval_nyuv2_depth.py", parse_nyuv2_depth_args),
            ("eval_navi_depth.py", parse_navi_depth_args),
            ("eval_nyuv2_normals.py", parse_nyuv2_normals_args),
            ("eval_navi_normals.py", parse_navi_normals_args),
        )
        for script, parser in entries:
            with self.subTest(script=script), unittest.mock.patch(
                "sys.argv",
                [script, "--model-id", "openai/clip-vit-large-patch14"],
            ):
                args = parser()
            self.assertEqual(args.torch_dtype, "bf16")
            self.assertEqual(args.seed, 42)
            self.assertFalse(hasattr(args, "dataset"))
            self.assertFalse(hasattr(args, "task"))
            self.assertFalse(hasattr(args, "processor_id"))

    def test_clip_siglip_geometry_paper_references(self) -> None:
        expected = {
            ("openai/clip-vit-large-patch14", "nyuv2", "depth"): ("depth_rmse", 0.553),
            ("openai/clip-vit-large-patch14", "navi", "depth"): ("depth_rmse", 0.073),
            ("google/siglip-so400m-patch14-384", "nyuv2", "depth"): ("depth_rmse", 0.563),
            ("google/siglip-so400m-patch14-384", "navi", "depth"): ("depth_rmse", 0.069),
            ("google/siglip2-so400m-patch14-384", "nyuv2", "depth"): ("depth_rmse", 0.466),
            ("google/siglip2-so400m-patch14-384", "navi", "depth"): ("depth_rmse", 0.064),
            ("openai/clip-vit-large-patch14", "nyuv2", "normals"): ("angular_rmse_degrees", 24.3),
            ("openai/clip-vit-large-patch14", "navi", "normals"): ("angular_rmse_degrees", 25.5),
            ("google/siglip-so400m-patch14-384", "nyuv2", "normals"): ("angular_rmse_degrees", 24.1),
            ("google/siglip-so400m-patch14-384", "navi", "normals"): ("angular_rmse_degrees", 25.4),
            ("google/siglip2-so400m-patch14-384", "nyuv2", "normals"): ("angular_rmse_degrees", 23.0),
            ("google/siglip2-so400m-patch14-384", "navi", "normals"): ("angular_rmse_degrees", 25.0),
        }
        for (model_id, dataset, task), (metric, value) in expected.items():
            with self.subTest(model_id=model_id, dataset=dataset, task=task):
                reference = paper_reference(model_id, dataset, task)
                self.assertIsNotNone(reference)
                self.assertEqual(reference["metric"], metric)
                self.assertEqual(reference["value"], value)
        self.assertIsNone(paper_reference("local/model", "nyuv2", "depth"))

    def test_clip_nyuv2_depth_paper_reference(self) -> None:
        reference = paper_reference("openai/clip-vit-large-patch14", "nyuv2", "depth")
        self.assertIsNotNone(reference)
        self.assertEqual(reference["metric"], "depth_rmse")
        self.assertEqual(reference["value"], 0.553)
        self.assertIn("CLIP L/14@224", reference["source"])

    def test_geonet_removes_the_two_corrupt_samples(self) -> None:
        raw = [f"aaa_{index:05d}.mat" for index in range(6_919)]
        raw.append(NYU_GEONET_BAD_FILES[0])
        raw.extend(f"ccc_{index:05d}.mat" for index in range(14_261))
        raw.append(NYU_GEONET_BAD_FILES[1])
        raw.extend(f"zzz_{index:05d}.mat" for index in range(9_734))

        cleaned, precleaned = clean_nyu_geonet_instances(raw)
        self.assertFalse(precleaned)
        self.assertEqual(len(cleaned), NYU_GEONET_USABLE_SAMPLES)
        self.assertTrue(set(cleaned).isdisjoint(NYU_GEONET_BAD_FILES))
        replayed, precleaned = clean_nyu_geonet_instances(cleaned)
        self.assertTrue(precleaned)
        self.assertEqual(replayed, cleaned)

    def test_frozen_features_remain_usable_by_trainable_heads(self) -> None:
        class DummyTower:
            device_name = "cpu"

            @staticmethod
            def normalize(images: torch.Tensor) -> torch.Tensor:
                return images

            @staticmethod
            def final_features(images: torch.Tensor) -> torch.Tensor:
                return images[:, :2]

            @staticmethod
            def dpt_features(images: torch.Tensor):
                patches = images[:, :2]
                global_token = patches.mean(dim=(2, 3))
                return [(global_token, patches) for _ in range(4)]

        tower = DummyTower()
        images = torch.randn(2, 3, 4, 4)
        final = extract_final_features(tower, images, 1, "fp32")
        dpt = extract_dpt_features(tower, images, 1, "fp32")
        self.assertFalse(final.is_inference())
        self.assertTrue(all(not tensor.is_inference() for pair in dpt for tensor in pair))

        head = torch.nn.Conv2d(2, 1, kernel_size=1)
        head(final).sum().backward()
        self.assertIsNotNone(head.weight.grad)

    def test_snapshot_revision_is_parsed_only_from_snapshot_paths(self) -> None:
        revision = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
        path = Path("/cache/models--google--siglip/snapshots") / revision / "config.json"
        self.assertEqual(_snapshot_revision_from_path(path), revision)
        self.assertIsNone(_snapshot_revision_from_path("/cache/config.json"))
        self.assertIsNone(_snapshot_revision_from_path(None))
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_cached_snapshot_revision(directory, "config.json"))

    def test_deterministic_batch_stream_resumes_exactly(self) -> None:
        full = list(DeterministicBatchSampler(11, 3, 8, 0, 10, True))
        prefix = list(DeterministicBatchSampler(11, 3, 8, 0, 4, True))
        suffix = list(DeterministicBatchSampler(11, 3, 8, 4, 6, True))
        self.assertEqual(full, prefix + suffix)

    def test_deterministic_augmentation_seed_is_replayable(self) -> None:
        class RandomDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                return index, random.random(), float(np.random.random()), float(torch.rand(()))

        dataset = DeterministicAugmentDataset(RandomDataset())
        self.assertEqual(dataset[(0, 1234)], dataset[(0, 1234)])

    def _dummy_tower(self, family: str) -> FrozenVisionTower:
        class AddConstant(torch.nn.Module):
            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return tokens + 100.0

        class DummyVision(torch.nn.Module):
            def __init__(self, model_family: str) -> None:
                super().__init__()
                self.model_family = model_family
                self.post_layernorm = AddConstant()

            def forward(self, pixel_values: torch.Tensor, **kwargs):
                del kwargs
                batch = pixel_values.shape[0]
                token_count = 5 if self.model_family == "clip" else 4
                hidden_states = []
                for layer in range(5):
                    values = torch.arange(token_count * 2, dtype=torch.float32)
                    values = values.view(1, token_count, 2).repeat(batch, 1, 1)
                    hidden_states.append(values + layer * 10.0)
                return SimpleNamespace(
                    hidden_states=tuple(hidden_states),
                    last_hidden_state=hidden_states[-1],
                    pooler_output=torch.full((batch, 2), 999.0),
                )

        tower = FrozenVisionTower.__new__(FrozenVisionTower)
        torch.nn.Module.__init__(tower)
        tower.metadata = VisionMetadata(
            model_id="dummy",
            processor_id="dummy",
            family=family,
            hidden_size=2,
            patch_size=1,
            num_hidden_layers=4,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
        tower.vision_model = DummyVision(family)
        return tower

    def test_clip_uses_layer_cls_while_siglip_uses_final_map(self) -> None:
        images = torch.zeros(1, 3, 2, 2)
        clip = self._dummy_tower("clip")
        clip_pairs = clip.dpt_features(images, layer_indices=(0, 1, 2, 3))
        for layer, (global_token, patch_map) in enumerate(clip_pairs, start=1):
            expected_cls = torch.tensor([[100.0 + layer * 10, 101.0 + layer * 10]])
            self.assertTrue(torch.equal(global_token, expected_cls))
            self.assertEqual(patch_map.shape, (1, 2, 2, 2))
        clip_final = clip.final_features(images)
        self.assertEqual(float(clip_final[0, 0, 0, 0]), 142.0)

        siglip = self._dummy_tower("siglip")
        siglip_pairs = siglip.dpt_features(images, layer_indices=(0, 1, 2, 3))
        for global_token, patch_map in siglip_pairs:
            self.assertTrue(torch.equal(global_token, torch.full((1, 2), 999.0)))
            self.assertEqual(patch_map.shape, (1, 2, 2, 2))

    def test_dpt_layers_are_uniform_quartiles(self) -> None:
        tower = self._dummy_tower("siglip")
        metadata = tower.metadata.__dict__
        tower.metadata = VisionMetadata(**{**metadata, "num_hidden_layers": 27})
        self.assertEqual(tower.default_dpt_layer_indices(), (5, 12, 19, 26))
        tower.metadata = VisionMetadata(**{**metadata, "num_hidden_layers": 40})
        self.assertEqual(tower.default_dpt_layer_indices(), (9, 19, 29, 39))

    def test_linear_depth_head_bounds_shape_and_gradient(self) -> None:
        head = LinearDepthHead(6, min_depth=0.001, max_depth=10.0, num_bins=8)
        self.assertEqual(head.classifier.kernel_size, (1, 1))
        self.assertEqual(head.patch_size, 14)
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in head.modules()))
        features = torch.randn(2, 6, 3, 4)
        with unittest.mock.patch(
            "eval.eval_nyuv2_depth.F.interpolate",
            wraps=torch.nn.functional.interpolate,
        ) as interpolate:
            prediction = head(features, image_size=(15, 19))
        self.assertEqual(interpolate.call_args_list[-1].args[1], (42, 56))
        self.assertEqual(prediction.shape, (2, 1, 15, 19))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.detach().min()), 0.001)
        self.assertLessEqual(float(prediction.detach().max()), 10.0)
        prediction.mean().backward()
        self.assertIsNotNone(head.classifier.weight.grad)

    def test_small_dpt_heads_shape_and_gradient(self) -> None:
        feature_pairs = [
            (torch.randn(2, 8), torch.randn(2, 8, 4, 4))
            for _ in range(4)
        ]
        common = {
            "input_embed_dim": 8,
            "channels": 8,
            "post_process_channels": (4, 8, 8, 16),
        }
        depth_head = DepthDecoder(num_depth_bins=8, **common)
        depth = depth_head(feature_pairs, image_size=(31, 29))
        self.assertEqual(depth.shape, (2, 1, 31, 29))
        self.assertTrue(torch.isfinite(depth).all())
        depth.mean().backward()
        self.assertIsNotNone(depth_head.head.weight.grad)

        normals_head = NormalsDecoder(**common)
        normals = normals_head(feature_pairs, image_size=(31, 29))
        self.assertEqual(normals.shape, (2, 3, 31, 29))
        self.assertTrue(torch.isfinite(normals).all())

    def test_tips_normal_resize_normalizes_before_and_after_bicubic(self) -> None:
        raw = torch.tensor(
            [[
                [[1.0, 2.0], [1.0, 2.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]],
            requires_grad=True,
        )
        with unittest.mock.patch(
            "eval.geometry_common.F.interpolate",
            wraps=torch.nn.functional.interpolate,
        ) as interpolate:
            prediction = normalize_and_resize_normals(raw, (7, 9))
        self.assertEqual(prediction.shape, (1, 3, 7, 9))
        self.assertEqual(interpolate.call_args.kwargs["mode"], "bicubic")
        self.assertFalse(interpolate.call_args.kwargs["align_corners"])
        lengths = torch.linalg.vector_norm(prediction, dim=1)
        self.assertTrue(torch.allclose(lengths, torch.ones_like(lengths), atol=1e-6))
        prediction.sum().backward()
        self.assertIsNotNone(raw.grad)
        with self.assertRaisesRegex(ValueError, "Bx3xHxW"):
            normalize_and_resize_normals(torch.ones(1, 4, 2, 2), (4, 4))

    def test_three_channel_angular_loss_uses_depth_valid_mask(self) -> None:
        target = torch.zeros(1, 3, 1, 2)
        target[:, 2] = 1.0
        prediction = target.clone()
        prediction[:, 2, 0, 1] = -1.0
        one_valid = torch.tensor([[[[1.0, 0.0]]]])
        both_valid = torch.ones_like(one_valid)
        masked = normal_loss(prediction, target, one_valid)
        unmasked = normal_loss(prediction, target, both_valid)
        self.assertLess(float(masked), float(unmasked))
        with self.assertRaisesRegex(ValueError, "three-channel"):
            normal_loss(torch.ones(1, 4, 1, 2), target, both_valid)

    def test_per_image_metrics_are_exact_for_perfect_predictions(self) -> None:
        depth = torch.tensor(
            [
                [[[1.0, 2.0], [0.0, 4.0]]],
                [[[2.0, 0.0], [3.0, 5.0]]],
            ]
        )
        depth_result = depth_metrics(depth, depth)
        self.assertTrue(torch.equal(depth_result["rmse"], torch.zeros(2)))
        self.assertTrue(torch.equal(depth_result["abs_rel"], torch.zeros(2)))
        self.assertTrue(torch.equal(depth_result["delta_1"], torch.ones(2)))

        normals = torch.zeros(2, 3, 2, 2)
        normals[:, 2] = 1.0
        normal_result = normal_metrics(normals, normals, depth)
        self.assertTrue(torch.allclose(normal_result["rmse"], torch.zeros(2)))
        self.assertTrue(torch.equal(normal_result["delta_1"], torch.ones(2)))

    def test_nyuv2_depth_metrics_apply_eigen_crop(self) -> None:
        target = torch.ones(1, 1, 480, 640)
        prediction = torch.full_like(target, 10.0)
        prediction[..., 45:471, 41:601] = 1.0
        cropped = depth_metrics(prediction, target, nyu_crop=True)
        full = depth_metrics(prediction, target)
        self.assertEqual(float(cropped["rmse"][0]), 0.0)
        self.assertGreater(float(full["rmse"][0]), 0.0)

    def test_nyuv2_depth_metrics_exclude_range_boundaries(self) -> None:
        target = torch.zeros(1, 1, 480, 640)
        prediction = torch.ones_like(target)
        target[..., 100, 100:105] = torch.tensor([1.0, 0.001, 10.0, 0.0005, 11.0])
        prediction[..., 100, 101:105] = torch.tensor([10.0, 1.0, 10.0, 1.0])
        metrics = depth_metrics(prediction, target, nyu_crop=True)
        self.assertEqual(float(metrics["rmse"][0]), 0.0)
        self.assertEqual(float(metrics["abs_rel"][0]), 0.0)
        self.assertEqual(float(metrics["delta_1"][0]), 1.0)

    def test_nyuv2_depth_train_transform_preserves_tips_resolution(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
        transformed_image, transformed_depth = nyu_depth_train_transform(
            image,
            depth,
            augment=False,
        )
        self.assertEqual(transformed_image.shape, (480, 640, 3))
        self.assertEqual(transformed_depth.shape, (480, 640))
        np.testing.assert_array_equal(transformed_image, image)
        np.testing.assert_array_equal(transformed_depth, depth)

    def test_probe3d_learning_rate_boundaries(self) -> None:
        values = (1000, 5e-4, 100, 0.01, 0.01)
        self.assertAlmostEqual(learning_rate_at_step(0, *values), 5e-6)
        self.assertAlmostEqual(learning_rate_at_step(100, *values), 5e-4)
        self.assertAlmostEqual(learning_rate_at_step(1000, *values), 5e-6)

    def test_navi_depth_defaults_combine_tips_and_probe3d(self) -> None:
        limits = run_limits(False)
        self.assertEqual(limits.steps, 50_000)
        self.assertEqual(limits.batch_size, 8)
        self.assertEqual(NAVI_DEPTH_LEARNING_RATE, 5e-4)
        self.assertEqual(NAVI_DEPTH_WARMUP_STEPS, 7_500)
        self.assertEqual(NAVI_DEPTH_SIG_LOSS_WEIGHT, 10.0)
        self.assertEqual(NAVI_DEPTH_GRADIENT_LOSS_WEIGHT, 0.5)

    def test_normal_defaults_follow_tips_steps_and_probe3d_optimizer(self) -> None:
        limits = run_limits(False)
        self.assertEqual(limits.steps, 50_000)
        self.assertEqual(limits.batch_size, 8)
        self.assertEqual(NYUV2_NORMALS_LEARNING_RATE, 5e-4)
        self.assertEqual(NAVI_NORMALS_LEARNING_RATE, 5e-4)
        self.assertEqual(NYUV2_NORMALS_WARMUP_STEPS, 7_500)
        self.assertEqual(NAVI_NORMALS_WARMUP_STEPS, 7_500)

    def test_navi_normals_keep_metric_depth_for_the_valid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(data_root=root)
            limits = RunLimits(
                steps=2,
                batch_size=2,
                eval_interval=1,
                save_interval=1,
                log_interval=1,
                max_train=3,
                max_eval=2,
            )
            train_dataset = object()
            eval_dataset = object()
            with unittest.mock.patch(
                "eval.eval_navi_normals.NAVIProbeDataset",
                side_effect=[train_dataset, eval_dataset],
            ) as dataset_class:
                built_train, built_eval = build_navi_normals_datasets(args, limits)
        self.assertIs(built_train, train_dataset)
        self.assertIs(built_eval, eval_dataset)
        self.assertFalse(dataset_class.call_args_list[0].kwargs["relative_depth"])
        self.assertFalse(dataset_class.call_args_list[1].kwargs["relative_depth"])

    def test_dinov2_linear_depth_learning_rate_boundaries(self) -> None:
        values = (50_000, 1e-4, 12_800, 1e-8, 0.001)
        self.assertAlmostEqual(learning_rate_at_step(0, *values), 1e-7)
        self.assertAlmostEqual(learning_rate_at_step(12_800, *values), 1e-4)
        self.assertAlmostEqual(learning_rate_at_step(50_000, *values), 1e-12)

    def test_dinov2_one_cycle_beta1_boundaries(self) -> None:
        self.assertAlmostEqual(one_cycle_beta1_at_step(0, 50_000), 0.95)
        self.assertAlmostEqual(one_cycle_beta1_at_step(14_999, 50_000), 0.85)
        self.assertAlmostEqual(one_cycle_beta1_at_step(49_999, 50_000), 0.95)

    def test_sigloss_distinguishes_dinov2_and_probe3d_variance(self) -> None:
        prediction = torch.tensor([[[[1.0, 2.0]]]])
        target = torch.ones_like(prediction)
        unbiased = sig_loss(prediction, target, 1.0, False, True)
        biased = sig_loss(prediction, target, 1.0, False, False)
        self.assertGreater(float(unbiased), float(biased))

    def test_gradient_loss_downsamples_spatial_axes_and_is_batch_order_invariant(self) -> None:
        target = torch.ones(2, 1, 12, 12)
        prediction = target.clone()
        prediction[1, :, :, 6:] = 2.0
        loss = gradient_loss(prediction, target)
        flipped = gradient_loss(prediction.flip(0), target.flip(0))
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.allclose(loss, flipped))

    def test_navi_disparity_conversion(self) -> None:
        disparity = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            Image.fromarray(disparity).save(path)
            depth = read_navi_depth(path)
        self.assertEqual(float(depth[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(depth[0, 0, 1]), 0.01, places=6)
        self.assertAlmostEqual(float(depth[0, 1, 0]), 0.0199997, places=5)

    def test_navi_relative_depth_matches_probe3d(self) -> None:
        depth = torch.tensor([[[0.0, 0.2, 0.4, 0.6]]])
        normalized = normalize_navi_relative_depth(depth, torch.tensor(0.2))
        expected = torch.tensor([[[0.0, 0.01, 0.505, 1.0]]])
        self.assertTrue(torch.allclose(normalized, expected))

    def test_navi_crop_and_camera_normals(self) -> None:
        image = torch.rand(3, 8, 8)
        depth = torch.zeros(1, 8, 8)
        depth[:, 1:6, 2:7] = 1.0
        image_crop, depth_crop = bbox_crop(image, depth)
        self.assertEqual(image_crop.shape[-2:], depth_crop.shape[-2:])
        self.assertEqual(depth_crop.shape[-2:], (4, 4))

        flat_depth = torch.ones(1, 8, 8)
        normals = depth_to_normals(flat_depth, focal_length=500.0)
        interior_norm = torch.linalg.vector_norm(normals[:, 1:-1, 1:-1], dim=0)
        self.assertTrue(torch.allclose(interior_norm, torch.ones_like(interior_norm)))
        self.assertEqual(float(normals[:, 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
