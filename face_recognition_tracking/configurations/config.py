from typing import Tuple

EMBEDDING_MODEL_IMAGE_SHAPE: Tuple[int, int] = (224, 224)
VECTORDB_COLLECTION_NAME: str = "image_searchs"
WHITELIST_IMAGE_EXTENSIONS: list[str] = [".jpg", ".png", ".jpeg", ".webp"]
CHROMADB_PERSISTENT_DIRECTORY: str = "chromadb"
THRESHOLD_MATCHED_FACE: int = 8083
