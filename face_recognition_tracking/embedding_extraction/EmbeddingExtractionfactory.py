from face_recognition_tracking.configurations.config import (
    VGG_EMBEDDING_EXTRACTOR,
    CUSTOM_EMBEDDING_EXTRACTOR,
)
from face_recognition_tracking.embedding_extraction.VGGEmbeddingExtractor import (
    VGGEmbeddingExtractor,
)


class EmbeddingFactory:

    @staticmethod
    def create_embedding_extractor(
        detector_type: str = VGG_EMBEDDING_EXTRACTOR,
    ):  # using haarcascades detector in default
        if detector_type == VGG_EMBEDDING_EXTRACTOR:
            return VGGEmbeddingExtractor()
        elif detector_type == CUSTOM_EMBEDDING_EXTRACTOR:
            # TODO need to implement openface face detector
            pass
        else:
            raise ValueError(f"Unknown detector type: {detector_type} provided.")
