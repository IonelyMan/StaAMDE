from phase2.image.models import build_convnextv2, convnextv2_names


def available_models():
    return convnextv2_names()


def build_model(arch: str, num_classes: int = 2, drop_path_rate: float = 0.1):
    if arch.startswith("convnextv2_"):
        return build_convnextv2(arch, num_classes=num_classes, drop_path_rate=drop_path_rate)
    raise ValueError(f"未知图像模型: {arch}，可选: {', '.join(available_models())}")
