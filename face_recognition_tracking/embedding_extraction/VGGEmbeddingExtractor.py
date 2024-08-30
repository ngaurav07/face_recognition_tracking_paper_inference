from typing import List

import keras
import numpy as np

from face_recognition_tracking.configurations.config import EMBEDDING_MODEL_IMAGE_SHAPE
from face_recognition_tracking.embedding_extraction.BaseExtractor import BaseExtractor
from face_recognition_tracking.utils import ImageHelper


class VGGEmbeddingExtractor(BaseExtractor):

    embedding_model = None

    def __init__(self):
        # Load the VGG16 model without the top classification layer
        base_model = keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

        # Add a Global Average Pooling layer to convert feature maps to a single vector
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)  # 128-dimensional embedding

        # Create the embedding model
        VGGEmbeddingExtractor.embedding_model = keras.models.Model(
            inputs=base_model.input, outputs=x
        )

    def extract_embedding(self, image: np.ndarray) -> List[float]:
        """
        Extract the embedding of the image.
        Preprocess the image and return the extracted embedding.
        :param image: numpy ndarray
        :return: embedding of the image
        """
        processed_image = self._preprocess(image)
        embeddings = VGGEmbeddingExtractor.embedding_model.predict(processed_image)
        return embeddings

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image and make it ready for embedding model.
        Resize the image as per requirement by the model. Ex (224,224)
        Add new dimension to the image.
        :param image:  numpy ndarray image
        :return: pre-processed numpy ndarray image
        """
        image = ImageHelper.resize_image(image, EMBEDDING_MODEL_IMAGE_SHAPE)
        image = np.expand_dims(image, axis=0)
        image = keras.applications.vgg16.preprocess_input(image)
        return image
