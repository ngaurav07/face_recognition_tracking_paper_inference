import os
import uuid
from typing import Dict, Any

import cv2

from face_recognition_tracking.configurations.config import WHITELIST_IMAGE_EXTENSIONS
from face_recognition_tracking.embedding_extraction.extract import EmbeddingExtractor
from face_recognition_tracking.search import VectorDatabase
from face_recognition_tracking.utils import load_image_from_path

embedding = EmbeddingExtractor()
vectorDb = VectorDatabase()


def webcam_inference():
    # load the video frame
    cap = cv2.VideoCapture(0)  # using webcam for now

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        # get the faces from the image

        # search for the images

        # track the image

        # show in the whole tracked image
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if not ret:
            print("Cannot receive image. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


def save_images_embedding(image_dir: str):
    for root, directory, files in os.walk(
        image_dir
    ):  # walk in the images directory in every folder
        for file in files[:2]:  # only take 2 images per person
            image_path = os.path.join(root, file)
            extension = os.path.splitext(image_path)[1]
            if extension in WHITELIST_IMAGE_EXTENSIONS:  # only take images file
                image = load_image_from_path(image_path)
                image_embedding = embedding.extract(image)
                folder_name = os.path.basename(root)
                metadata: Dict[str, Any] = {
                    "person_name": folder_name,
                    "file_name": file,
                }
                vectorDb.save_embedding(
                    image_embedding, metadata, uuid.uuid4()
                )  # save images embedding in the vector database


def image_inference(image_path: str):
    # load the image
    image = load_image_from_path(image_path)
    image_embedding = embedding.extract(image)

    # save or extract image embedding
    return vectorDb.match_face(image_embedding)
