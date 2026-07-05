import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

IMAGE_MODELS: Dict[str, Dict[str, str]] = {
    "clip_base": {"model_id": "openai/clip-vit-large-patch14", "processor_id": "openai/clip-vit-large-patch14", "family": "clip_l14_224", "stage": "baseline", "label": "CLIP-L/14-224 baseline"},
    "clip_s1": {"model_id": "fesvhtr/clip-r-s1-run1207-1280", "processor_id": "openai/clip-vit-large-patch14", "family": "clip_l14_224", "stage": "s1", "label": "CLIP-L/14-224 S1"},
    "clip_s2": {"model_id": "fesvhtr/clip-r-s2-run1219-505", "processor_id": "openai/clip-vit-large-patch14", "family": "clip_l14_224", "stage": "s2", "label": "CLIP-L/14-224 S2"},
    "siglip_base": {"model_id": "google/siglip-so400m-patch14-384", "processor_id": "google/siglip-so400m-patch14-384", "family": "siglip_so400m_384", "stage": "baseline", "label": "SigLIP-So400M/14-384 baseline"},
    "siglip_s1": {"model_id": "fesvhtr/siglip-r-s1-run0201-1280", "processor_id": "google/siglip-so400m-patch14-384", "family": "siglip_so400m_384", "stage": "s1", "label": "SigLIP-So400M/14-384 S1"},
    "siglip_s2": {"model_id": "fesvhtr/siglip-r-s2-run0203-673", "processor_id": "google/siglip-so400m-patch14-384", "family": "siglip_so400m_384", "stage": "s2", "label": "SigLIP-So400M/14-384 S2"},
}

DEFAULT_PROMPTS = [
    {"id": "unstable_leaning", "text": "something appears unstable because it is leaning or only partly supported"},
    {"id": "weather_protection", "text": "an object is being used to protect someone from weather"},
    {"id": "preparing_action", "text": "a person is preparing to perform an action"},
    {"id": "carrying_serving", "text": "objects are arranged to make carrying or serving easier"},
    {"id": "spilling_overflowing", "text": "a liquid or material appears to be spilling or overflowing"},
    {"id": "support_surface", "text": "an object is supported by a flat surface underneath it"},
    {"id": "containment", "text": "something is being held inside a container or enclosed space"},
    {"id": "damaged_missing", "text": "an object looks damaged incomplete or missing a part"},
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def first_sentence(text: str, max_words: int = 32) -> str:
    sentences = split_sentences(text)
    if sentences:
        return truncate_words(sentences[0], max_words)
    return truncate_words(normalize_text(text), max_words)


def truncate_words(text: str, max_words: int = 40) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "."


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def parse_model_keys(models: str | None) -> List[str]:
    if not models or models == "all":
        return list(IMAGE_MODELS.keys())
    keys = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in keys if m not in IMAGE_MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(IMAGE_MODELS)}")
    return keys


def save_numpy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))


def topk_indices(sim: np.ndarray, k: int, exclude_self: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("topk_indices expects a square matrix")
    n = sim.shape[0]
    k = max(1, min(k, n - 1 if exclude_self else n))
    work = np.array(sim, copy=True)
    if exclude_self:
        np.fill_diagonal(work, -np.inf)
    part = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(work, part, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(scores, order, axis=1)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.pad(coords, ((0, 0), (0, 1)))
    coords = coords[:, :2].astype(np.float32)
    scale = np.percentile(np.abs(coords), 99, axis=0)
    coords = coords / np.maximum(scale, 1e-6)
    return np.clip(coords, -1.5, 1.5)


def simple_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def seeded_rows(rows: Sequence[Dict[str, Any]], limit: int | None, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows
