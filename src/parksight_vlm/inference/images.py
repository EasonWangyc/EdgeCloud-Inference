"""共享单图预处理；模型运行时的 processor 仍负责 patch 与张量转换。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parksight_vlm.workload import FrozenWorkload

from .runtime import RuntimeDependencyError, RuntimeInputError


def load_workload_image(image_path: Path, workload: FrozenWorkload) -> Any:
    """转换为 RGB，并按冻结 workload 的宽高进行 bicubic 缩放。"""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeDependencyError("Pillow is required for workload image resizing") from error
    try:
        with Image.open(image_path) as source:
            return source.convert("RGB").resize(
                (workload.input_size.width, workload.input_size.height),
                resample=Image.Resampling.BICUBIC,
            )
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise RuntimeInputError(f"cannot decode image {image_path.name}: {error}") from error
