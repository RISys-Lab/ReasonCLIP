#!/usr/bin/env python3
import argparse
from typing import Dict, List, Tuple

from common import normalize_text, read_jsonl, split_sentences, truncate_words, write_jsonl

TYPE_RULES: Dict[str, Dict[str, object]] = {
    "support": {
        "keywords": [" on ", " atop ", "underneath", "under ", "resting", "laid", "sitting", "standing", "leaning", "stacked", "touching", "supported", "balanced"],
        "conclusion": "one object is supported, positioned, or constrained by another visible object",
        "weight": 1.25,
    },
    "containment": {
        "keywords": ["inside", "within", "filled", "contains", "holding", "basket", "bowl", "box", "cup", "bag", "container", "in a ", "in an "],
        "conclusion": "something is contained, carried, or held by another object",
        "weight": 1.15,
    },
    "protection": {
        "keywords": ["umbrella", "helmet", "glove", "jacket", "mask", "cover", "covered", "shelter", "shield"],
        "conclusion": "an object is likely serving as protection or coverage",
        "weight": 1.45,
    },
    "use_or_function": {
        "keywords": ["handle", "wheel", "sign", "plate", "table", "chair", "tray", "cart", "tool", "utensil", "screen", "bench", "shelf"],
        "conclusion": "a visible object is serving a practical function in the scene",
        "weight": 1.0,
    },
    "state_or_activity": {
        "keywords": ["parked", "walking", "running", "riding", "cutting", "preparing", "open", "closed", "missing", "half", "broken", "cracked", "blank", "worn", "lying", "sleeping"],
        "conclusion": "the object or person is in a specific visible state or activity",
        "weight": 1.55,
    },
    "spatial_relation": {
        "keywords": ["behind", "in front", "left", "right", "above", "below", "near", "between", "surrounding", "perpendicular", "horizontal", "vertical", "along", "to the"],
        "conclusion": "the positions of the visible objects matter for interpreting the scene",
        "weight": 0.65,
    },
    "material_state": {
        "keywords": ["spilling", "overflowing", "wet", "dry", "smooth", "rough", "dirty", "rust", "rusted", "melted", "smoke", "steam", "liquid", "dried", "weathered", "patina"],
        "conclusion": "a visible material or surface condition changes how the scene should be interpreted",
        "weight": 1.5,
    },
    "affordance": {
        "keywords": ["road", "street", "curb", "seat", "stairs", "ladder", "path", "grip", "strap", "door"],
        "conclusion": "the visible structure affords a likely use or action",
        "weight": 0.95,
    },
}


def score_sentence(sentence: str) -> Tuple[float, str]:
    lower = f" {sentence.lower()} "
    best_type = "other"
    best = 0.0
    for typ, rule in TYPE_RULES.items():
        count = sum(1 for kw in rule["keywords"] if kw in lower)
        weighted = count * float(rule.get("weight", 1.0))
        if weighted > best:
            best = weighted
            best_type = typ
    length_bonus = 0.25 if 6 <= len(sentence.split()) <= 35 else 0.0
    return best * 3.0 + length_bonus, best_type


def make_reasoning_item(sentence: str, reasoning_type: str) -> Dict[str, str]:
    sentence = normalize_text(sentence).rstrip(" .")
    sentence_short = truncate_words(sentence, 34).rstrip(" .")
    if reasoning_type not in TYPE_RULES:
        reasoning_type = "other"
    rule = TYPE_RULES.get(reasoning_type, {})
    conclusion = str(rule.get("conclusion", "the visible evidence supports a grounded commonsense interpretation"))
    evidence = sentence_short[0].lower() + sentence_short[1:] if sentence_short else "visible details are described"
    caption = f"{sentence_short}. This visible evidence suggests that {conclusion}."
    return {
        "caption": caption,
        "visible_evidence": evidence,
        "commonsense_conclusion": conclusion,
        "reasoning_type": reasoning_type,
        "source_sentence": sentence,
    }


def select_diverse(ranked, max_captions: int):
    selected = []
    used_types = set()
    seen_sentences = set()
    for pass_idx in (0, 1):
        for score, _neg_idx, typ, sentence in ranked:
            if score <= 0 and selected:
                continue
            key = normalize_text(sentence).lower()
            if key in seen_sentences:
                continue
            if pass_idx == 0 and typ in used_types:
                continue
            selected.append((typ, sentence))
            used_types.add(typ)
            seen_sentences.add(key)
            if len(selected) >= max_captions:
                return selected
    return selected


def annotate_record(row: Dict[str, object], max_captions: int) -> Dict[str, object]:
    source_caption = str(row.get("source_caption") or "")
    sentences = split_sentences(source_caption)
    if not sentences:
        sentences = [str(row.get("descriptive_caption") or "visible details in the scene")]
    ranked = []
    for idx, sentence in enumerate(sentences):
        score, typ = score_sentence(sentence)
        ranked.append((score, -idx, typ, sentence))
    ranked.sort(reverse=True)
    selected = select_diverse(ranked, max_captions)
    if not selected:
        selected = [("other", sentences[0])]
    out = dict(row)
    out["reasoning_captions"] = [make_reasoning_item(sentence, typ) for typ, sentence in selected]
    out["annotation_method"] = "docci_text_rules_v2_evidence_first"
    out["annotation_model"] = "codex_text_heuristic"
    out["annotation_prompt_version"] = "reasoning_caption_v1"
    return out


def main():
    parser = argparse.ArgumentParser(description="Create visually grounded reasoning captions from DOCCI long descriptions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-captions", type=int, default=3)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    annotated = [annotate_record(row, args.max_captions) for row in rows]
    write_jsonl(args.output, annotated)
    n_caps = sum(len(r.get("reasoning_captions", [])) for r in annotated)
    print(f"wrote {len(annotated)} annotated rows / {n_caps} reasoning captions to {args.output}")


if __name__ == "__main__":
    main()
