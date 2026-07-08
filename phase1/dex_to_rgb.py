import argparse
import csv
import os
import struct
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

"""
将 APK 内的 classes*.dex 转换为图像。

重点：当 --color-mode rgb 时，RGB 图生成逻辑与给出的 Java 代码保持一致：
1. 对 APK 内所有 classes*.dex，分别提取六类 DEX 索引表：
   string_ids / type_ids / proto_ids / field_ids / method_ids / class_defs。
2. 先按类别跨 DEX 拼接，再按类别顺序合并：
   all_string -> all_type -> all_proto -> all_field -> all_method -> all_class。
3. 每 3 个字节映射为 1 个 RGB 像素。
4. 图片宽度固定 512。
5. 图片高度为 ceil(total_pixels / 512)。
6. 不做 resize，不固定 256x256。
7. height <= 100 时不保存图像，与 Java 中 return 的逻辑一致。

其他输入方式、输出目录、train/val/test 与 mal/benign 划分逻辑保持不变。
"""

VALID_SPLITS = {"train", "val", "test"}
LABEL_DIR_ALIASES = {
    1: ("mal", "malware", "1"),
    0: ("benign", "leg", "normal", "0"),
}

JAVA_RGB_IMAGE_WIDTH = 512
JAVA_RGB_MIN_HEIGHT_EXCLUSIVE = 100


def get_image_width(file_size_bytes: int) -> int:
    """Nataraj malware image 系列工作常用的二进制尺寸启发式宽度。仅用于非 rgb 灰度伪彩色模式。"""
    size_kb = file_size_bytes / 1024
    if size_kb < 10:
        return 32
    if size_kb < 30:
        return 64
    if size_kb < 60:
        return 128
    if size_kb < 100:
        return 256
    if size_kb < 200:
        return 384
    if size_kb < 500:
        return 512
    if size_kb < 1000:
        return 768
    return 1024


def label_name(label: int) -> str:
    return "mal" if int(label) == 1 else "benign"


def normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None or split == "":
        return None
    value = split.strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(f"belong/split 只能是 {sorted(VALID_SPLITS)}，当前为: {split}")
    return value


def output_parent(output_root: Path, split: Optional[str], label: int) -> Path:
    if split:
        return output_root / split / label_name(label)
    return output_root / label_name(label)


def read_u32_le(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from("<I", data, offset)[0]


def safe_slice(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size <= 0 or offset >= len(data):
        return b""
    end = min(len(data), offset + size)
    return data[offset:end]


def parse_dex_sections(dex_bytes: bytes) -> List[Tuple[str, bytes]]:
    """解析 DEX section。用于非 RGB 模式或兼容旧 source-mode。"""
    if len(dex_bytes) < 112 or not dex_bytes.startswith(b"dex\n"):
        return [("raw_dex", dex_bytes)]

    section_specs = [
        ("header", 0, 112),
        ("string_ids", read_u32_le(dex_bytes, 0x3C), read_u32_le(dex_bytes, 0x38) * 4),
        ("type_ids", read_u32_le(dex_bytes, 0x44), read_u32_le(dex_bytes, 0x40) * 4),
        ("proto_ids", read_u32_le(dex_bytes, 0x4C), read_u32_le(dex_bytes, 0x48) * 12),
        ("field_ids", read_u32_le(dex_bytes, 0x54), read_u32_le(dex_bytes, 0x50) * 8),
        ("method_ids", read_u32_le(dex_bytes, 0x5C), read_u32_le(dex_bytes, 0x58) * 8),
        ("class_defs", read_u32_le(dex_bytes, 0x64), read_u32_le(dex_bytes, 0x60) * 32),
        ("data", read_u32_le(dex_bytes, 0x6C), read_u32_le(dex_bytes, 0x68)),
    ]

    sections = []
    for name, offset, size in section_specs:
        chunk = safe_slice(dex_bytes, offset, size)
        if chunk:
            sections.append((name, chunk))
    return sections or [("raw_dex", dex_bytes)]


def extract_java_style_dex_indexes(dex_bytes: bytes) -> Tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
    """
    按 Java 代码的方式提取六类索引表。

    Java 代码没有提取 header，也没有提取 data 区；只提取：
    string_ids / type_ids / proto_ids / field_ids / method_ids / class_defs。
    """
    if len(dex_bytes) < 112 or not dex_bytes.startswith(b"dex\n"):
        return b"", b"", b"", b"", b"", b""

    string_size = read_u32_le(dex_bytes, 0x38) * 4
    string_off = read_u32_le(dex_bytes, 0x3C)

    type_size = read_u32_le(dex_bytes, 0x40) * 4
    type_off = read_u32_le(dex_bytes, 0x44)

    proto_size = read_u32_le(dex_bytes, 0x48) * 12
    proto_off = read_u32_le(dex_bytes, 0x4C)

    field_size = read_u32_le(dex_bytes, 0x50) * 8
    field_off = read_u32_le(dex_bytes, 0x54)

    method_size = read_u32_le(dex_bytes, 0x58) * 8
    method_off = read_u32_le(dex_bytes, 0x5C)

    class_size = read_u32_le(dex_bytes, 0x60) * 32
    class_off = read_u32_le(dex_bytes, 0x64)

    return (
        safe_slice(dex_bytes, string_off, string_size),
        safe_slice(dex_bytes, type_off, type_size),
        safe_slice(dex_bytes, proto_off, proto_size),
        safe_slice(dex_bytes, field_off, field_size),
        safe_slice(dex_bytes, method_off, method_size),
        safe_slice(dex_bytes, class_off, class_size),
    )


def build_java_style_rgb_stream(dex_chunks: List[bytes]) -> bytes:
    """
    与 Java mergeDexIndexes(String[] dexPaths) 等价：
    先把所有 DEX 的同类 section 分别拼接，最后再按类别顺序合并。
    """
    string_data: List[bytes] = []
    type_data: List[bytes] = []
    proto_data: List[bytes] = []
    field_data: List[bytes] = []
    method_data: List[bytes] = []
    class_data: List[bytes] = []

    for dex_bytes in dex_chunks:
        string_part, type_part, proto_part, field_part, method_part, class_part = extract_java_style_dex_indexes(dex_bytes)
        string_data.append(string_part)
        type_data.append(type_part)
        proto_data.append(proto_part)
        field_data.append(field_part)
        method_data.append(method_part)
        class_data.append(class_part)

    return b"".join([
        b"".join(string_data),
        b"".join(type_data),
        b"".join(proto_data),
        b"".join(field_data),
        b"".join(method_data),
        b"".join(class_data),
    ])


def build_dex_stream(dex_chunks: List[bytes], source_mode: str) -> bytes:
    """旧逻辑保留给 viridis/jet 等非 rgb 模式使用。"""
    pieces: List[bytes] = []
    for dex_bytes in dex_chunks:
        sections = parse_dex_sections(dex_bytes)
        if source_mode == "full":
            pieces.append(dex_bytes)
        elif source_mode == "indexes":
            pieces.extend(chunk for name, chunk in sections if name in {
                "header",
                "string_ids",
                "type_ids",
                "proto_ids",
                "field_ids",
                "method_ids",
                "class_defs",
            })
        else:
            pieces.extend(chunk for _, chunk in sections)
    return b"".join(pieces)


def byte_stream_to_raw_gray(dex_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    file_size = len(dex_bytes)
    width = get_image_width(file_size)
    arr = np.frombuffer(dex_bytes, dtype=np.uint8)
    remainder = file_size % width
    if remainder:
        arr = np.pad(arr, (0, width - remainder), mode="constant", constant_values=0)
    height = len(arr) // width
    return arr.reshape((height, width)), width, height


def byte_stream_to_fixed_square_gray(dex_bytes: bytes, target_size: int) -> np.ndarray:
    """将完整字节序列压缩/补齐为固定正方形单通道图像。仅用于非 rgb 模式。"""
    target_pixels = target_size * target_size
    arr = np.frombuffer(dex_bytes, dtype=np.uint8)
    if len(arr) == target_pixels:
        fixed = arr
    elif len(arr) > target_pixels:
        fixed = cv2.resize(
            arr.reshape(1, -1),
            (target_pixels, 1),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1)
    else:
        fixed = np.pad(arr, (0, target_pixels - len(arr)), mode="constant", constant_values=0)
    return fixed.astype(np.uint8).reshape((target_size, target_size))


def byte_stream_to_rgb_pixels_java(data: bytes) -> np.ndarray:
    """
    与 Java 代码一致：totalPixels = allData.length / 3。
    注意：Java 是整数除法，最后不足 3 字节的尾部会被丢弃，不补 0。
    """
    total_pixels = len(data) // 3
    if total_pixels <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    usable = data[:total_pixels * 3]
    return np.frombuffer(usable, dtype=np.uint8).reshape((total_pixels, 3)).astype(np.uint8)


def rgb_pixels_to_java_raw_image(data: bytes, width: int = JAVA_RGB_IMAGE_WIDTH) -> Optional[np.ndarray]:
    """
    Java RGB 图尺寸逻辑：
    width = 512
    height = ceil(totalPixels / 512)
    height <= 100 时直接不生成图。
    """
    pixels = byte_stream_to_rgb_pixels_java(data)
    total_pixels = len(pixels)
    if total_pixels <= 0:
        return None

    height = int(np.ceil(total_pixels / width))
    # if height <= JAVA_RGB_MIN_HEIGHT_EXCLUSIVE:
    #     return None

    target_pixels = height * width
    if total_pixels < target_pixels:
        pad = np.zeros((target_pixels - total_pixels, 3), dtype=np.uint8)
        pixels = np.vstack([pixels, pad])

    return pixels.reshape((height, width, 3))


def normalize_contrast(img_gray: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    low, high = np.percentile(img_gray, [low_percentile, high_percentile])
    if high <= low:
        return img_gray.astype(np.uint8)
    clipped = np.clip(img_gray.astype(np.float32), low, high)
    normalized = (clipped - low) * (255.0 / (high - low))
    return normalized.astype(np.uint8)


def to_rgb(img_gray: np.ndarray, color_mode: str) -> np.ndarray:
    if color_mode == "viridis":
        return cv2.applyColorMap(img_gray, cv2.COLORMAP_VIRIDIS)
    if color_mode == "jet":
        return cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    raise ValueError(f"unsupported color mode for gray image: {color_mode}")


def build_non_rgb_image(
    dex_stream: bytes,
    shape_mode: str,
    target_size: int,
    color_mode: str,
    contrast: bool,
    denoise: str,
) -> np.ndarray:
    """保留原来的灰度伪彩色图逻辑。"""
    raw_gray, _, _ = byte_stream_to_raw_gray(dex_stream)

    if shape_mode == "raw":
        img_gray = raw_gray
    elif shape_mode == "resized_raw":
        img_gray = cv2.resize(raw_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        img_gray = byte_stream_to_fixed_square_gray(dex_stream, target_size)

    if contrast:
        img_gray = normalize_contrast(img_gray)

    if denoise == "median":
        img_gray = cv2.medianBlur(img_gray, 3)
    elif denoise == "gaussian":
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    return to_rgb(img_gray, color_mode)


def process_single_apk(task: Tuple[str, int, Optional[str], str, bool, str, str, str, int, bool, str]) -> Tuple[bool, str, str]:
    apk_text, label, split, output_text, overwrite, color_mode, source_mode, shape_mode, target_size, contrast, denoise = task
    apk_path = Path(apk_text)
    out_dir = output_parent(Path(output_text), split, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / f"{apk_path.stem}.png"

    try:
        if out_img_path.exists() and not overwrite:
            return True, apk_path.name, "skipped"

        dex_chunks: List[bytes] = []
        with zipfile.ZipFile(apk_path, "r") as apk_zip:
            dex_files = sorted(name for name in apk_zip.namelist() if name.endswith(".dex"))
            if not dex_files:
                return False, apk_path.name, "no dex"
            for dex_file in dex_files:
                dex_chunks.append(apk_zip.read(dex_file))

        if color_mode == "rgb":
            # RGB 模式完全采用 Java 代码的索引区合并 + 固定宽 512 + 高度自适应逻辑。
            rgb_stream = build_java_style_rgb_stream(dex_chunks)
            if len(rgb_stream) == 0:
                return False, apk_path.name, "empty dex indexes"

            img_rgb = rgb_pixels_to_java_raw_image(rgb_stream, width=JAVA_RGB_IMAGE_WIDTH)
            if img_rgb is None:
                return False, apk_path.name, f"height <= {JAVA_RGB_MIN_HEIGHT_EXCLUSIVE} or empty image"

            # 为了尽量等价 Java，不建议对 RGB 图做去噪；但如果命令行显式指定，仍然保留原参数能力。
            if denoise == "median":
                img_rgb = cv2.medianBlur(img_rgb, 3)
            elif denoise == "gaussian":
                img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)

            cv2.imwrite(str(out_img_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            return True, apk_path.name, str(out_img_path)

        # 非 RGB 模式沿用旧逻辑。
        dex_stream = build_dex_stream(dex_chunks, source_mode)
        if len(dex_stream) == 0:
            return False, apk_path.name, "empty dex"

        img = build_non_rgb_image(
            dex_stream=dex_stream,
            shape_mode=shape_mode,
            target_size=target_size,
            color_mode=color_mode,
            contrast=contrast,
            denoise=denoise,
        )
        cv2.imwrite(str(out_img_path), img)
        return True, apk_path.name, str(out_img_path)

    except zipfile.BadZipFile:
        return False, apk_path.name, "bad zip"
    except Exception as exc:
        return False, apk_path.name, str(exc)


def iter_apks(sample_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in sample_dir.rglob("*.apk") if path.is_file())


def infer_label_dir(dataset_root: Path, split: str, label: int) -> Optional[Path]:
    for alias in LABEL_DIR_ALIASES[int(label)]:
        candidate = dataset_root / split / alias
        if candidate.exists():
            return candidate
    return None


def load_manifest(manifest_path: Path, default_split: Optional[str]) -> List[Tuple[Path, int, Optional[str]]]:
    samples = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"apk_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest CSV 必须包含 apk_path,label 两列")
        for row in reader:
            row_split = row.get("split") or row.get("belong") or default_split
            samples.append((Path(row["apk_path"]), int(row["label"]), normalize_split(row_split)))
    return samples


def collect_samples(args: argparse.Namespace) -> List[Tuple[Path, int, Optional[str]]]:
    samples: List[Tuple[Path, int, Optional[str]]] = []
    default_split = normalize_split(args.belong)

    if args.manifest:
        samples.extend(load_manifest(Path(args.manifest), default_split))

    if args.dataset_root:
        if default_split is None:
            raise ValueError("使用 --dataset-root 时必须传入 --belong train|val|test")
        dataset_root = Path(args.dataset_root)
        for label in [1, 0]:
            sample_dir = infer_label_dir(dataset_root, default_split, label)
            if sample_dir is None:
                print(f"[warning] 未找到 {dataset_root / default_split} 下 label={label} 的目录")
                continue
            samples.extend((apk_path, label, default_split) for apk_path in iter_apks(sample_dir))

    for spec in args.input:
        if "=" not in spec:
            raise ValueError("--input 格式应为 label=目录，例如 1=/data/mal")
        label_text, dir_text = spec.split("=", 1)
        sample_dir = Path(dir_text)
        if not sample_dir.exists():
            print(f"[warning] 目录不存在，跳过: {sample_dir}")
            continue
        samples.extend((apk_path, int(label_text), default_split) for apk_path in iter_apks(sample_dir))

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 APK 中的 DEX 转换为 RGB 图像")
    parser.add_argument("--input", action="append", default=[], help="label=目录，可重复传入")
    parser.add_argument("--dataset-root", help="原始数据根目录，自动读取 dataset_root/belong/mal 与 dataset_root/belong/benign")
    parser.add_argument("--belong", choices=sorted(VALID_SPLITS), help="当前处理的数据划分: train、val 或 test")
    parser.add_argument("--manifest", help="CSV，包含 apk_path,label 两列；可选 split 或 belong 列")
    parser.add_argument("--output", required=True, help="图像输出目录")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "viridis", "jet"],
        default="rgb",
        help="rgb=使用 Java 代码同款 DEX 索引区 RGB 图；viridis/jet=保留旧灰度伪彩色逻辑",
    )
    parser.add_argument(
        "--source-mode",
        choices=["structured", "indexes", "full"],
        default="structured",
        help="仅对 viridis/jet 生效；rgb 模式固定使用 Java 同款六类索引表",
    )
    parser.add_argument(
        "--shape-mode",
        choices=["fixed_square", "resized_raw", "raw"],
        default="fixed_square",
        help="仅对 viridis/jet 生效；rgb 模式固定宽 512、高度自适应、不 resize",
    )
    parser.add_argument("--target-size", type=int, default=256, help="仅对 viridis/jet 生效；rgb 模式忽略")
    parser.add_argument("--contrast", action="store_true", help="仅对 viridis/jet 生效；rgb 模式不做对比度拉伸")
    parser.add_argument(
        "--denoise",
        choices=["none", "median", "gaussian"],
        default="none",
        help="可选去噪；为了等价 Java，rgb 模式建议保持 none",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = [
        (
            str(apk_path),
            label,
            split,
            args.output,
            args.overwrite,
            args.color_mode,
            args.source_mode,
            args.shape_mode,
            args.target_size,
            args.contrast,
            args.denoise,
        )
        for apk_path, label, split in collect_samples(args)
        if apk_path.exists() and zipfile.is_zipfile(apk_path)
    ]

    print(f"Total APKs found: {len(samples)}")
    if not samples:
        return

    start_time = time.time()
    success_count = 0
    fail_count = 0
    workers = max(1, min(args.workers, len(samples)))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_apk, sample): sample for sample in samples}
        for index, future in enumerate(as_completed(futures), 1):
            success, apk_name, message = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"[{index}/{len(samples)}] [-] {apk_name}: {message}")
            if index % 50 == 0 or index == len(samples):
                print(f"[{index}/{len(samples)}] success={success_count}, failed={fail_count}")

    elapsed = time.time() - start_time
    print("-" * 30)
    print("Processing Complete")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Speed: {len(samples) / max(elapsed, 1e-9):.2f} APK/s")


if __name__ == "__main__":
    main()
