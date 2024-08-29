from typing import Tuple

import cv2
import numpy as np

from face_recognition_tracking.configurations import embedding_model_image_shape


def load_image_from_path(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def resize_image(image: np.ndarray, size: Tuple[int, int] = embedding_model_image_shape) -> np.ndarray:
    return cv2.resize(image, size)
