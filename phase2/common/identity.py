from typing import Optional


def normalize_split(split: Optional[str]) -> str:
    return split if split else "unknown"


def build_sample_key(sample_id: str, split: Optional[str], label: Optional[int]) -> str:
    label_text = "unknown" if label is None else str(int(label))
    return f"{normalize_split(split)}:{label_text}:{sample_id}"
