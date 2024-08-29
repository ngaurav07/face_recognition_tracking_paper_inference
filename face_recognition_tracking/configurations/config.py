from typing import Tuple

EMBEDDING_MODEL_IMAGE_SHAPE: Tuple[int, int] = (224, 224)
VECTORDB_COLLECTION_NAME: str = "image_searchs"
WHITELIST_IMAGE_EXTENSIONS: list[str] = [".jpg", ".png", ".jpeg", ".webp"]
CHROMADB_PERSISTENT_DIRECTORY: str = "chromadb"
THRESHOLD_MATCHED_FACE: int = 8083
HAARCASCADES_FACE_DETECTOR: str = "HAARCASCADES_FACE_DETECTOR"
OPENFACE_FACE_DETECTOR: str = "OPENFACE_FACE_DETECTOR"


# Detector
BOUNDING_BOXES_FOR_FACES: str = "bounding_boxes"
FACE_FRAME: str = "face_frame"
