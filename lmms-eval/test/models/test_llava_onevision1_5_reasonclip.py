import unittest

from lmms_eval.models.simple.llava_onevision1_5 import (
    Llava_OneVision1_5,
    _format_multimodal_content,
    apply_reasonclip_chat_template,
)


class TestReasonCLIPChatTemplate(unittest.TestCase):
    def test_preserves_multimodal_content(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Follow the instructions."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": object()},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]

        self.assertEqual(
            apply_reasonclip_chat_template(messages),
            (
                "<|im_start|>system\nFollow the instructions.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>Describe this image."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        )

    def test_uses_training_system_prompt(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        self.assertTrue(
            apply_reasonclip_chat_template(messages).startswith(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            )
        )

    def test_rejects_unknown_content(self):
        with self.assertRaisesRegex(ValueError, "Unsupported message content type"):
            _format_multimodal_content([{"type": "audio", "audio": object()}])

    def test_single_process_model_is_not_unwrapped(self):
        class SingleProcessAccelerator:
            num_processes = 1

            def unwrap_model(self, model):
                raise AssertionError("unwrap_model was called")

        model = object()
        adapter = object.__new__(Llava_OneVision1_5)
        adapter._model = model
        adapter.accelerator = SingleProcessAccelerator()

        self.assertIs(adapter.model, model)
