from typing import Tuple
from copy import deepcopy

import numpy as np


def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...], target_length) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes: np.ndarray, original_size: Tuple[int, ...], target_length) -> np.ndarray:
    """
    Expects a numpy array shape Bx4. Requires the original image size
    in (H, W) format.
    """
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, target_length)
    return boxes.reshape(-1, 4)


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)
