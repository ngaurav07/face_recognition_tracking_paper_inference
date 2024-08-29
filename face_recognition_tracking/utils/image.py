from typing import Tuple

import cv2
import numpy as np

from face_recognition_tracking.configurations import EMBEDDING_MODEL_IMAGE_SHAPE


def load_image_from_path(image_path: str) -> np.ndarray:
    """

    Args:
        image_path:

    Returns:

    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def resize_image(
    image: np.ndarray, size: Tuple[int, int] = EMBEDDING_MODEL_IMAGE_SHAPE
) -> np.ndarray:
    """

    Args:
        image:
        size:
    """
    return cv2.resize(image, size)
