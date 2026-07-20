import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABEL_ALIASES = {
    0: ("benign", "leg", "normal", "0"),
    1: ("mal", "malware", "1"),
}


@dataclass
class ImageSample:
    image_path: Path
    label: int
    split: str

    @property
    def sample_id(self) -> str:
        return self.image_path.stem

    @property
    def apk_name(self) -> str:
        return f"{self.sample_id}.apk"


def resolve_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if len(image_size) == 1:
        return int(image_size[0]), int(image_size[0])
    return int(image_size[0]), int(image_size[1])


def discover_samples(input_dir: Path, split: str) -> List[ImageSample]:
    split_dir = input_dir / split if input_dir.name.lower() != split else input_dir
    samples: List[ImageSample] = []
    seen = set()
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            class_dir = split_dir / alias
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    samples.append(ImageSample(path, int(label), split))
    if not samples:
        raise ValueError(f"没有在 {split_dir} 下找到图像样本，请检查目录是否为 {split}/mal 和 {split}/benign")
    return samples


class DexImageDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        split: str,
        image_size: Sequence[int] | int = (224, 224),
        train: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.split = split
        self.image_size = resolve_image_size(image_size)
        self.train = train
        self.samples = discover_samples(self.input_dir, split)

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_info(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "apk_name": sample.apk_name,
            "split": sample.split,
            "label": sample.label,
            "image_path": str(sample.image_path),
        }

    def _read_rgb(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _augment_train(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = self.image_size
        scale = random.uniform(1.0, 1.15)
        resize_h = max(target_h, int(round(target_h * scale)))
        resize_w = max(target_w, int(round(target_w * scale)))
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        top = random.randint(0, max(0, resize_h - target_h))
        left = random.randint(0, max(0, resize_w - target_w))
        image = image[top : top + target_h, left : left + target_w]
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
        if random.random() < 0.35:
            factor = random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return image

    def _resize_eval(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = self.image_size
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(image))

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._read_rgb(sample.image_path)
        image = self._augment_train(image) if self.train else self._resize_eval(image)
        return self._to_tensor(image), int(sample.label)
