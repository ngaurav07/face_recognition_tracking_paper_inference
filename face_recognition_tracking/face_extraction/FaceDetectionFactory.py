from face_recognition_tracking.configurations.config import (
    HAARCASCADES_FACE_DETECTOR,
    OPENFACE_FACE_DETECTOR,
)
from face_recognition_tracking.face_extraction.HaarcascadesFaceDetector import (
    HaarcascadesFaceDetector,
)


class FaceDetectionFactory:

    @staticmethod
    def create_face_detector(
        detector_type: str = HAARCASCADES_FACE_DETECTOR,
    ):  # using haarcascades detector in default
        if detector_type == HAARCASCADES_FACE_DETECTOR:
            return HaarcascadesFaceDetector()
        elif detector_type == OPENFACE_FACE_DETECTOR:
            # TODO need to implement openface face detector
            pass
        else:
            raise ValueError(f"Unknown detector type: {detector_type} provided.")
