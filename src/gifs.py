"""Write rollout frames to a GIF with Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 120) -> None:
    if not frames:
        raise ValueError("cannot write a GIF with no frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
