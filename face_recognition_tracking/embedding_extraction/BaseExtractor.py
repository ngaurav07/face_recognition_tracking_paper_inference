from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseExtractor(ABC):
    @abstractmethod
    def extract_embedding(self, image: np.ndarray) -> List[float]:
        """
        Method to extract embedding of face from the face of image.
        Args:
            image: The input image from which embedding needs to be extracted

        Returns:
            Embedding of the provided image

        """
        pass
