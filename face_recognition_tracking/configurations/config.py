from typing import Tuple

from typing_extensions import Final

EMBEDDING_MODEL_IMAGE_SHAPE: Final[Tuple[int, int]] = (224, 224)
VECTORDB_COLLECTION_NAME: Final[str] = "face_recognition"
WHITELIST_IMAGE_EXTENSIONS: Final[list[str]] = [".jpg", ".png", ".jpeg", ".webp"]
CHROMADB_PERSISTENT_DIRECTORY: Final[str] = "chromadb"
THRESHOLD_MATCHED_FACE: Final[int] = 8083

# Face Detector
HAARCASCADES_FACE_DETECTOR: Final[str] = "HAARCASCADES_FACE_DETECTOR"
OPENFACE_FACE_DETECTOR: Final[str] = "OPENFACE_FACE_DETECTOR"
BOUNDING_BOXES_FOR_FACES: Final[str] = "bounding_boxes"
FACE_FRAME: Final[str] = "face_frame"

# Embedding Extractor
VGG_EMBEDDING_EXTRACTOR: Final[str] = "VGG_EMBEDDING_EXTRACTOR"
CUSTOM_EMBEDDING_EXTRACTOR: Final[str] = "CUSTOM_EMBEDDING_EXTRACTOR"
