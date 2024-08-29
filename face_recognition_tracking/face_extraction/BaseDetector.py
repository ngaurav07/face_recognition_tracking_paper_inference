from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect_faces(self, frame) -> List[np.ndarray]:
        """
        Method to detect faces from the frame.
        Args:
            frame: The input image in which faces needs to be detected.

        Returns:
            A list of frame of  detected faces
        """
        pass
