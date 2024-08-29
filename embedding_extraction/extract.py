import keras
import numpy as np

from utils import resize_image


class EmbeddingExtractor():
    embedding_model = None
    def __init__(self):
        # Load the VGG16 model without the top classification layer
        base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Add a Global Average Pooling layer to convert feature maps to a single vector
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)  # 128-dimensional embedding

        # Create the embedding model
        EmbeddingExtractor.embedding_model = keras.models.Model(inputs=base_model.input, outputs=x)

    def extract(self, image: np.ndarray):
        processed_image = self._preprocess(image)
        embeddings = EmbeddingExtractor.embedding_model.predict(processed_image)
        return embeddings

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = resize_image(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = keras.applications.vgg16.preprocess_input(image)
        return image
