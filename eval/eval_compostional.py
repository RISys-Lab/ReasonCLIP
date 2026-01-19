import argparse
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


def compute_similarity(model, processor, images, text, device):
    inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.image_embeds @ outputs.text_embeds.t()


def get_image_to_text_score(model, processor, images, text, device, return_tot=False):
    similarity_scores = compute_similarity(model, processor, images, text, device)
    if not return_tot:
        return int(similarity_scores.argmax() == 0), similarity_scores

    i0_c0 = similarity_scores[0, 0].item()
    i0_c1 = similarity_scores[0, 1].item()
    i0_c2 = similarity_scores[0, 2].item()
    image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_embeds = model.get_text_features(**inputs)
    text_similarity_scores = text_embeds @ text_embeds.t()
    c0_c1 = text_similarity_scores[0, 1].item()
    c0_c2 = text_similarity_scores[0, 2].item()
    c1_c2 = text_similarity_scores[1, 2].item()
    text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

    return int(image_correct), similarity_scores, int(text_correct), [c0_c1, c0_c2, c1_c2]


def eval_whatsup(model, processor, device):
    dataset = load_dataset("Mayfull/whats_up_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in dataset:
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device)
        if sample["original_file_name"].startswith("coco"):
            name = "coco"
        elif sample["original_file_name"].startswith("vg"):
            name = "vg"
        else:
            name = "whatsup"
        result[name].append(score)
    average_result = {k: 100 * sum(v) / len(v) for k, v in result.items()}
    average_result["total"] = sum(average_result.values()) / len(average_result)
    return average_result


def eval_valse(model, processor, device):
    dataset = load_dataset("Mayfull/valse_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in dataset:
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device)
        result[sample["linguistic_phenomena"]].append(score)
        result["total"].append(score)
    average_result = {k: 100 * sum(v) / len(v) for k, v in result.items()}
    return average_result


def eval_crepe(model, processor, device):
    dataset = load_dataset("Mayfull/crepe_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in dataset:
        images = [
            i.crop(
                (
                    sample["x"],
                    sample["y"],
                    sample["x"] + sample["width"],
                    sample["y"] + sample["height"],
                )
            )
            for i in sample["images"]
        ]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device)
        result[sample["original_file_name"]].append(score)
    average_result = {k: 100 * sum(v) / len(v) for k, v in result.items()}
    average_result["total"] = sum(average_result.values()) / len(average_result)
    return average_result


def eval_sugarcrepe(model, processor, device):
    dataset = load_dataset("Mayfull/sugarcrepe_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in dataset:
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device)
        result[sample["original_file_name"]].append(score)
    average_result = {k: 100 * sum(v) / len(v) for k, v in result.items()}
    average_result["total"] = sum(average_result.values()) / len(average_result)
    return average_result


def eval_sugarcrepe_pp(model, processor, device):
    dataset = load_dataset("Mayfull/sugarcrepepp_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in dataset:
        images = sample["images"]
        text = [*sample["positive_caption_1"], *sample["positive_caption_2"], *sample["negative_caption"]]
        itt_score, _, tot_score, _ = get_image_to_text_score(model, processor, images, text, device, return_tot=True)
        result[sample["original_file_name"] + "_itt"].append(itt_score)
        result[sample["original_file_name"] + "_tot"].append(tot_score)
    average_result = {k: 100 * sum(v) / len(v) for k, v in result.items()}
    average_result["total_itt"] = sum(v for k, v in average_result.items() if k.endswith("itt")) / sum(
        k.endswith("itt") for k in average_result
    )
    average_result["total_tot"] = sum(v for k, v in average_result.items() if k.endswith("tot")) / sum(
        k.endswith("tot") for k in average_result
    )
    return average_result


def parse_args():
    parser = argparse.ArgumentParser(description="READ-CLIP compositional evaluation")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    results = {
        "WhatsUp": eval_whatsup(model, processor, device),
        "VALSE": eval_valse(model, processor, device),
        "CREPE": eval_crepe(model, processor, device),
        "SugarCrepe": eval_sugarcrepe(model, processor, device),
        "SugarCrepe++": eval_sugarcrepe_pp(model, processor, device),
    }

    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()