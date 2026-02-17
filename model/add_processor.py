from transformers import AutoModel, AutoProcessor

MODEL_ID = "fesvhtr/clip-r-336-s1-run1215-1280"
PROCESSOR_ID = "openai/clip-vit-large-patch14-336"

print(f"Loading model: {MODEL_ID}")
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading processor: {PROCESSOR_ID}")
processor = AutoProcessor.from_pretrained(PROCESSOR_ID, trust_remote_code=True)

print(f"Pushing model + processor to: {MODEL_ID}")
model.push_to_hub(MODEL_ID, safe_serialization=True)
processor.push_to_hub(MODEL_ID)

print("Done.")
