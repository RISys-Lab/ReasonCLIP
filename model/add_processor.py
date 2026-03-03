from transformers import AutoProcessor

MODEL_ID = "fesvhtr/clip-r-336-s2-run0204-505"
PROCESSOR_ID = "openai/clip-vit-large-patch14-336"

processor = AutoProcessor.from_pretrained(PROCESSOR_ID, trust_remote_code=True)
processor.push_to_hub(MODEL_ID)
print("processor uploaded.")