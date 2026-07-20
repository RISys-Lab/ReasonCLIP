#!/usr/bin/env python3
"""Focused regression tests for the LocCa grounding protocol."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from eval.grounding import (
    C4Tokenizer,
    LocCaDecoder,
    box_iou_xyxy,
    box_string,
    decoder_loss,
    dequantize_box_lbrt,
    parse_box_string,
    prompt_string,
    quantize_box_xywh,
    shift_right,
    target_string,
)
from eval.grounding_data import (
    EpochGroundingRecordSampler,
    GroundingRecord,
    image_to_tensor,
    prompt_tokens,
    training_tokens,
)
from eval.eval_grounding import (
    adafactor_parameter_groups,
    cosine_learning_rate,
    file_fingerprint,
    grounding_implementation_assumptions,
    grounding_training_steps,
    paper_baseline_for_run,
    parse_args,
    protocols_compatible_for_resume,
    resume_epoch_position,
)
from eval.probe_utils import FrozenVisionTower, VisionMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = (
    REPO_ROOT
    / "data"
    / "downstream_data"
    / "C4Tokenizer"
    / "cc_en.32000.sentencepiece.model"
)


class GroundingProtocolTest(unittest.TestCase):
    def test_grounding_training_step_accounting(self) -> None:
        self.assertEqual(grounding_training_steps(321_327, 512, 50), (627, 31_350))
        self.assertEqual(grounding_training_steps(321_327, 256, 50), (1_255, 62_750))
        with self.assertRaises(ValueError):
            grounding_training_steps(3, 4, 50)

    def test_paper_baseline_requires_complete_baseline_run(self) -> None:
        args = SimpleNamespace(
            training_mix="full",
            loss_scope="full_aref",
            epochs=50,
            max_train=None,
            max_eval=None,
        )
        model_id = "google/siglip-so400m-patch14-384"
        self.assertIsNotNone(
            paper_baseline_for_run(args, model_id, 384, 31_350, 31_350)
        )
        self.assertIsNone(
            paper_baseline_for_run(args, model_id, 384, 18_810, 31_350)
        )
        self.assertIsNone(
            paper_baseline_for_run(
                args,
                "fesvhtr/siglip-r-s2-run0203-673",
                384,
                31_350,
                31_350,
            )
        )

    def test_grounding_resize_matches_tf_bilinear_uint8_semantics(self) -> None:
        values = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        image = Image.fromarray(np.repeat(values[:, :, None], 3, axis=2))
        actual = image_to_tensor(image, resolution=3)
        expected_uint8 = torch.tensor(
            [[0, 50, 100], [100, 138, 177], [200, 227, 255]],
            dtype=torch.float32,
        )
        expected = expected_uint8.unsqueeze(0).repeat(3, 1, 1).div(255.0)
        self.assertTrue(torch.equal(actual, expected))

    def test_cli_defaults_match_reported_rec_table(self) -> None:
        with patch("sys.argv", ["eval_grounding.py", "--model-id", "dummy"]):
            args = parse_args()
        self.assertEqual(args.training_mix, "full")
        self.assertEqual(args.loss_scope, "full_aref")

    def test_big_vision_schedule_uses_zero_based_update_index(self) -> None:
        self.assertEqual(cosine_learning_rate(0, 100, 10, 3e-4), 0.0)
        self.assertAlmostEqual(cosine_learning_rate(1, 100, 10, 3e-4), 3e-5)
        self.assertAlmostEqual(cosine_learning_rate(10, 100, 10, 3e-4), 3e-4)
        self.assertGreater(cosine_learning_rate(99, 100, 10, 3e-4), 0.0)
        self.assertEqual(cosine_learning_rate(0, 100, 0, 3e-4), 3e-4)

    def test_record_fingerprint_covers_path_size_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            payload = b"reasonclip\n"
            path.write_bytes(payload)
            fingerprint = file_fingerprint(path)
            self.assertEqual(fingerprint["path"], str(path.resolve()))
        self.assertEqual(fingerprint["bytes"], len(payload))
        self.assertEqual(fingerprint["sha256"], hashlib.sha256(payload).hexdigest())

    def test_implementation_assumptions_report_actual_values(self) -> None:
        disclosure = grounding_implementation_assumptions(256, 0.1)
        self.assertIn("effective batch 256", disclosure)
        self.assertIn("warmup ratio 0.1", disclosure)
        self.assertNotIn("batch 512", disclosure)

    def test_resume_compatibility_ignores_only_nonbehavioral_metadata(self) -> None:
        saved = {
            "name": "grounding",
            "loss_scope": "full_aref",
            "paper_disclosure": {"note": "old"},
            "record_files": {
                "train_clean": {"sha256": "train"},
                "manifest": {"sha256": "old"},
            },
        }
        manifest_only = {
            "name": "grounding",
            "loss_scope": "full_aref",
            "paper_disclosure": {"note": "new"},
            "record_files": {
                "train_clean": {"sha256": "train"},
                "manifest": {"sha256": "new"},
            },
        }
        changed_records = {
            "name": "grounding",
            "loss_scope": "full_aref",
            "paper_disclosure": {"note": "new"},
            "record_files": {
                "train_clean": {"sha256": "changed"},
                "manifest": {"sha256": "new"},
            },
        }
        self.assertTrue(protocols_compatible_for_resume(saved, manifest_only))
        self.assertFalse(protocols_compatible_for_resume(saved, changed_records))
        changed_behavior = {**saved, "loss_scope": "box_suffix"}
        self.assertFalse(protocols_compatible_for_resume(saved, changed_behavior))

    def test_resume_compatibility_ignores_decoder_target_descriptions(self) -> None:
        saved = {
            "decoder_target": {
                "sequence": "ARef: expression : box",
                "teacher_forcing": True,
                "loss_scope": "box_suffix",
                "conditional_box_suffix_default": "old description",
                "full_aref_pretraining_diagnostic": "old description",
            },
            "record_files": {},
        }
        expected = {
            "decoder_target": {
                "sequence": "ARef: expression : box",
                "teacher_forcing": True,
                "loss_scope": "box_suffix",
                "conditional_box_suffix_diagnostic": "new description",
                "full_aref_default": "new description",
            },
            "record_files": {},
        }
        self.assertTrue(protocols_compatible_for_resume(saved, expected))
        expected["decoder_target"]["sequence"] = "changed"
        self.assertFalse(protocols_compatible_for_resume(saved, expected))

    def test_resume_position_does_not_repeat_completed_microbatches(self) -> None:
        self.assertEqual(resume_epoch_position(0, 11, 8), (0, 0))
        self.assertEqual(resume_epoch_position(7, 11, 8), (0, 56))
        self.assertEqual(resume_epoch_position(11, 11, 8), (1, 0))
        self.assertEqual(resume_epoch_position(29, 11, 8), (2, 56))

    def test_image_epoch_sampler_is_deterministic_and_covers_candidates(self) -> None:
        def record(image_id: int, ann_id: int, sentence_id: int) -> GroundingRecord:
            return GroundingRecord(
                dataset="refcoco",
                split="train",
                split_by="unc",
                ref_id=ann_id,
                sentence_id=sentence_id,
                ann_id=ann_id,
                image_id=image_id,
                file_name=f"{image_id}.jpg",
                width=500,
                height=500,
                bbox_xywh=(20.0, 40.0, 130.0, 440.0),
                expression=f"object {sentence_id}",
            )

        records = [
            record(10, 101, 1),
            record(10, 101, 2),
            record(10, 102, 3),
            record(20, 201, 4),
        ]
        pair_sampler = EpochGroundingRecordSampler(records, seed=7)
        self.assertEqual(len(pair_sampler), 4)
        self.assertEqual(
            pair_sampler.summary(),
            {
                "unit": "referring_sentence",
                "examples_per_epoch": 4,
                "candidate_sentences": 4,
                "candidate_annotations": 3,
                "unique_images": 2,
                "selection": "all referring sentences",
            },
        )
        self.assertEqual(pair_sampler.records_for_epoch(37), records)

        sampler = EpochGroundingRecordSampler(
            records,
            sample_one_per_image=True,
            seed=7,
        )
        self.assertEqual(len(sampler), 2)
        self.assertEqual(
            sampler.summary(),
            {
                "unit": "unique_image",
                "examples_per_epoch": 2,
                "candidate_sentences": 4,
                "candidate_annotations": 3,
                "unique_images": 2,
                "selection": (
                    "shuffled no-replacement annotation cycle per image, then uniform "
                    "sentence with replacement"
                ),
            },
        )

        for cycle_start in range(0, 20, 2):
            annotation_ids = {
                sampler.records_for_epoch(epoch)[0].ann_id
                for epoch in range(cycle_start, cycle_start + 2)
            }
            self.assertEqual(annotation_ids, {101, 102})

        choices = set()
        for epoch in range(128):
            selected = sampler.records_for_epoch(epoch)
            self.assertEqual([item.image_id for item in selected], [10, 20])
            choices.add((selected[0].ann_id, selected[0].sentence_id))
        self.assertEqual(choices, {(101, 1), (101, 2), (102, 3)})

        sampler.set_epoch(37)
        before_resume = [sampler.record_at(index) for index in range(len(sampler))]
        resumed = EpochGroundingRecordSampler(
            list(reversed(records)),
            sample_one_per_image=True,
            seed=7,
        )
        resumed.set_epoch(37)
        after_resume = [resumed.record_at(index) for index in range(len(resumed))]
        self.assertEqual(before_resume, after_resume)

        limited = EpochGroundingRecordSampler(
            records,
            sample_one_per_image=True,
            seed=7,
            max_images=1,
        )
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited.summary()["candidate_sentences"], 3)


    def test_aref_conditional_prompt_format(self) -> None:
        box = (20, 480, 150, 40)
        self.assertEqual(prompt_string("a puffin"), "aref: a puffin : ")
        self.assertEqual(
            target_string("a puffin", box),
            "aref: a puffin : [20, 480, 150, 40]",
        )

    def _dummy_tower(self, family: str, patch_size: int = 1) -> FrozenVisionTower:
        class AddConstant(torch.nn.Module):
            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return tokens + 100.0

        class DummyVision(torch.nn.Module):
            def __init__(self, model_family: str, patch: int) -> None:
                super().__init__()
                self.model_family = model_family
                self.patch = patch
                self.post_layernorm = AddConstant()

            def forward(self, pixel_values: torch.Tensor, **kwargs):
                del kwargs
                batch = pixel_values.shape[0]
                patch_count = (pixel_values.shape[-2] // self.patch) * (
                    pixel_values.shape[-1] // self.patch
                )
                token_count = patch_count + int(self.model_family == "clip")
                tokens = torch.arange(token_count * 2, dtype=torch.float32)
                tokens = tokens.view(1, token_count, 2).repeat(batch, 1, 1)
                return SimpleNamespace(last_hidden_state=tokens)

        tower = FrozenVisionTower.__new__(FrozenVisionTower)
        torch.nn.Module.__init__(tower)
        tower.metadata = VisionMetadata(
            model_id="dummy",
            processor_id="dummy",
            family=family,
            hidden_size=2,
            patch_size=patch_size,
            num_hidden_layers=1,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
        tower.vision_model = DummyVision(family, patch_size)
        return tower

    def test_method_figure_coordinate_order_and_round_trip(self) -> None:
        # LocCa's method figure shows this object as [left, bottom, right, top].
        quantized = quantize_box_xywh((20, 40, 130, 440), width=500, height=500)
        self.assertEqual(quantized, (20, 480, 150, 40))
        self.assertEqual(box_string(quantized), "[20, 480, 150, 40]")
        self.assertEqual(
            dequantize_box_lbrt(quantized, width=500, height=500),
            (20.0, 40.0, 150.0, 480.0),
        )

    def test_box_parser_rejects_invalid_coordinates(self) -> None:
        self.assertEqual(parse_box_string("[20, 480, 150, 40]"), (20, 480, 150, 40))
        self.assertEqual(parse_box_string("answer: 20 480 150 40"), (20, 480, 150, 40))
        self.assertIsNone(parse_box_string("[20, 40, 150, 480]"))
        self.assertIsNone(parse_box_string("[20, 480, 501, 40]"))
        self.assertIsNone(parse_box_string("no box"))

    def test_iou_is_exact(self) -> None:
        self.assertEqual(box_iou_xyxy((0, 0, 2, 2), (0, 0, 2, 2)), 1.0)
        self.assertEqual(box_iou_xyxy((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)
        self.assertAlmostEqual(box_iou_xyxy((0, 0, 2, 2), (1, 1, 3, 3)), 1.0 / 7.0)

    def test_grounding_sequence_keeps_clip_cls_and_excludes_siglip_map(self) -> None:
        images = torch.zeros(1, 3, 2, 2)
        clip_tokens = self._dummy_tower("clip").sequence_features(images)
        self.assertEqual(clip_tokens.shape, (1, 5, 2))
        self.assertEqual(float(clip_tokens.min()), 100.0)
        self.assertEqual(float(clip_tokens.max()), 109.0)

        siglip_tokens = self._dummy_tower("siglip").sequence_features(images)
        self.assertEqual(siglip_tokens.shape, (1, 4, 2))
        self.assertEqual(float(siglip_tokens.min()), 0.0)
        self.assertEqual(float(siglip_tokens.max()), 7.0)

    def test_siglip_384_uses_paper_sequence_length_without_padding(self) -> None:
        images = torch.zeros(1, 3, 384, 384)
        siglip_tokens = self._dummy_tower("siglip", patch_size=14).sequence_features(images)
        clip_tokens = self._dummy_tower("clip", patch_size=14).sequence_features(
            torch.zeros(1, 3, 224, 224)
        )
        self.assertEqual(siglip_tokens.shape, (1, 729, 2))
        self.assertEqual(clip_tokens.shape, (1, 257, 2))

    def test_small_decoder_shape_and_gradient(self) -> None:
        decoder = LocCaDecoder(
            vision_size=24,
            vocab_size=100,
            max_length=16,
            hidden_size=32,
            num_heads=4,
            mlp_size=64,
            num_layers=2,
            dropout=0.0,
        )
        labels = torch.randint(1, 100, (2, 12))
        input_ids = shift_right(labels, pad_id=0)
        logits = decoder(torch.randn(2, 7, 24), input_ids, pad_id=0)
        self.assertEqual(logits.shape, (2, 12, 100))
        loss = decoder_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(decoder.output.weight.grad)
        self.assertGreater(float(decoder.output.weight.grad.abs().sum()), 0.0)

    def test_adafactor_decay_matches_big_vision_kernel_filter(self) -> None:
        decoder = LocCaDecoder(
            vision_size=24,
            vocab_size=100,
            max_length=16,
            hidden_size=32,
            num_heads=4,
            mlp_size=64,
            num_layers=2,
            dropout=0.0,
        )
        groups, names = adafactor_parameter_groups(decoder, weight_decay=1e-4)
        self.assertIn("output.weight", names["decay"])
        self.assertIn("layers.0.self_attention.in_proj_weight", names["decay"])
        self.assertIn("token_embedding.weight", names["no_decay"])
        self.assertIn("position_embedding", names["no_decay"])
        self.assertIn("final_norm.weight", names["no_decay"])
        self.assertIn("output.bias", names["no_decay"])
        self.assertEqual(groups[0]["weight_decay"], 1e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        grouped = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(list(decoder.parameters())))
        self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))

    @unittest.skipUnless(TOKENIZER_PATH.is_file(), "C4 tokenizer not downloaded")
    def test_c4_tokenizer_matches_big_vision_contract(self) -> None:
        tokenizer = C4Tokenizer(TOKENIZER_PATH, max_length=8)
        self.assertEqual(tokenizer.vocab_size, 32_000)
        self.assertEqual(tokenizer.pad_id, 0)
        self.assertEqual(tokenizer.eos_id, 1)
        self.assertEqual(tokenizer.model_path, TOKENIZER_PATH.resolve())
        self.assertEqual(
            tokenizer.model_sha256,
            "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec",
        )
        self.assertEqual(tokenizer.encode("ARef:"), tokenizer.encode("aref:"))
        sticky = tokenizer.encode_sticky("this sentence intentionally has many tokens")
        self.assertEqual(len(sticky), 8)
        non_padding = [token_id for token_id in sticky if token_id != tokenizer.pad_id]
        self.assertEqual(non_padding[-1], tokenizer.eos_id)


    @unittest.skipUnless(TOKENIZER_PATH.is_file(), "C4 tokenizer not downloaded")
    def test_pretraining_and_downstream_aref_loss_scopes(self) -> None:
        tokenizer = C4Tokenizer(TOKENIZER_PATH)
        record = GroundingRecord(
            dataset="refcoco",
            split="train",
            split_by="unc",
            ref_id=1,
            sentence_id=2,
            ann_id=3,
            image_id=4,
            file_name="unused.jpg",
            width=500,
            height=500,
            bbox_xywh=(20.0, 40.0, 130.0, 440.0),
            expression="the standing puffin",
        )
        prompts, _ = prompt_tokens([record], tokenizer)
        labels, mask, truncated = training_tokens(
            [record],
            [(20, 480, 150, 40)],
            tokenizer,
            loss_scope="full_aref",
        )
        prompt_length = len(prompts[0])
        task_prefix_length = len(tokenizer.encode("aref:", add_eos=False))
        self.assertEqual(truncated, 0)
        self.assertEqual(labels[0, :prompt_length].tolist(), prompts[0])
        self.assertFalse(mask[0, :task_prefix_length].any())
        self.assertTrue(mask[0, task_prefix_length:prompt_length].all())
        self.assertTrue(mask[0, prompt_length:].any())
        active_labels = labels[0][mask[0]].tolist()
        self.assertEqual(active_labels[-1], tokenizer.eos_id)

        suffix_labels, suffix_mask, suffix_truncated = training_tokens(
            [record],
            [(20, 480, 150, 40)],
            tokenizer,
        )
        self.assertEqual(suffix_truncated, 0)
        self.assertTrue(torch.equal(labels, suffix_labels))
        self.assertFalse(suffix_mask[0, :prompt_length].any())
        self.assertTrue(suffix_mask[0, prompt_length:].any())
        self.assertEqual(suffix_labels[suffix_mask][-1].item(), tokenizer.eos_id)


    @unittest.skipUnless(TOKENIZER_PATH.is_file(), "C4 tokenizer not downloaded")
    def test_overlong_expression_keeps_sticky_eos_supervision(self) -> None:
        tokenizer = C4Tokenizer(TOKENIZER_PATH, max_length=16)
        record = GroundingRecord(
            dataset="refcoco",
            split="train",
            split_by="unc",
            ref_id=1,
            sentence_id=2,
            ann_id=3,
            image_id=4,
            file_name="unused.jpg",
            width=500,
            height=500,
            bbox_xywh=(20.0, 40.0, 130.0, 440.0),
            expression="a very long referring expression " * 20,
        )
        prompts, prompt_truncated = prompt_tokens([record], tokenizer)
        labels, mask, box_truncated = training_tokens(
            [record],
            [(20, 480, 150, 40)],
            tokenizer,
            loss_scope="full_aref",
        )
        self.assertEqual(prompt_truncated, 1)
        self.assertEqual(len(prompts[0]), tokenizer.max_length - 1)
        self.assertEqual(box_truncated, 1)
        task_prefix_length = len(tokenizer.encode("aref:", add_eos=False))
        self.assertEqual(int(mask.sum()), tokenizer.max_length - task_prefix_length)
        self.assertEqual(labels[mask][-1].item(), tokenizer.eos_id)

        _, suffix_mask, _ = training_tokens(
            [record],
            [(20, 480, 150, 40)],
            tokenizer,
        )
        self.assertEqual(int(suffix_mask.sum()), 1)
        self.assertEqual(labels[suffix_mask][-1].item(), tokenizer.eos_id)


if __name__ == "__main__":
    unittest.main()
