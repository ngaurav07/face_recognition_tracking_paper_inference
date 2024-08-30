from typing import Tuple, TypedDict, List, Annotated

import cv2
import numpy as np

from face_recognition_tracking.configurations import (
    BOUNDING_BOXES_FOR_FACES,
    FACE_FRAME,
)
from face_recognition_tracking.face_extraction.BaseDetector import BaseDetector


class DetectedFaces(TypedDict):
    BOUNDING_BOXES_FOR_FACES: Annotated[
        Tuple[int, int, int, int], "Coordinates for the bounding boxes (x,y,w,h)"
    ]
    FACE_FRAME: Annotated[np.ndarray, "Image frame containing the detected faces"]


class HaarcascadesFaceDetector(BaseDetector):
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame) -> List[DetectedFaces]:
        """
        Detect faces from the frame using opencv
        Args:
            frame: Opencv numpy array which might contain faces

        Returns:
            List for the faces with bounding boxes as x,y,w,h and frame as ndarray
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = self.face_classifier.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        face_frames: List[DetectedFaces] | None = None
        for x, y, w, h in faces:
            # Extract the face using array slicing
            face_frame = frame[y : y + h, x : x + w]
            face_frames.append(
                {BOUNDING_BOXES_FOR_FACES: (x, y, w, h), FACE_FRAME: face_frame}
            )

        # Return the list of extracted face frames
        return face_frames
