from typing import Dict, Any, List
from uuid import UUID

import chromadb
import numpy as np
from chromadb import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

from face_recognition_tracking.configurations.config import (
    VECTORDB_COLLECTION_NAME,
    CHROMADB_PERSISTENT_DIRECTORY,
    THRESHOLD_MATCHED_FACE,
)


class VectorDatabase:
    """
    Vector Database helper which extract matched face, saved embedding of the face.
    """

    def __init__(self):
        # client for the chromadb. Used persistent client for saving in the disk rather than RAM.
        self._chroma_client = chromadb.PersistentClient(
            path=CHROMADB_PERSISTENT_DIRECTORY,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        # collection inside the chromadb which can be used to separate different projects inside the same tenant.
        self._collection = self._chroma_client.get_or_create_collection(
            name=VECTORDB_COLLECTION_NAME
        )

    def match_face(self, faces: List[np.ndarray]) -> List[Any]:
        """
        Extract the similar face from the previous one that is stored in the database.
        The index in the return list of the extracted face information is similar to the provided faces.
        Args:
            faces: faces that appear in one frame

        Returns: Matched faced persons name if any or None.

        """
        matched_faces = []
        for face in faces:
            matched_face = self._extract_matched_face(face)
            if (
                len(matched_face["distances"][0]) > 0
                and matched_face["distances"][0][0] <= THRESHOLD_MATCHED_FACE
            ):
                matched_faces.append(matched_face["metadatas"][0][0]["person_name"])
            else:
                matched_faces.append(None)  # append None to save the input array length
        return matched_faces

    def save_embedding(
        self, embedding: np.ndarray, metadata: Dict[str, Any], id: UUID
    ) -> None:
        """
        Save embedding and its metadata to the vector database
        Args:
            embedding:  Embedding vector to be stored.
            metadata: Metadata associated with the embedding
            id: UUID that can be used for mapping with the another dataset.
        """
        embedding = embedding.tolist()
        self._collection.add(
            embeddings=[embedding[0]], metadatas=[metadata], ids=[str(id)]
        )

    def _extract_matched_face(self, embedding: np.ndarray) -> Any:
        """
        Extract matched faces using embedding
        Args:
            embedding: embedding of the face that needs to be matched with database embedding

        Returns: Matched faces metadata with id and distance

        """
        return self._collection.query(
            query_embeddings=embedding.tolist(), n_results=1
        )  # extract only one face that is closer
