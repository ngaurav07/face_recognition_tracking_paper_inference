import os
import uuid
from typing import Dict, Any

import cv2

from face_recognition_tracking.configurations import (
    WHITELIST_IMAGE_EXTENSIONS,
    FACE_FRAME,
    BOUNDING_BOXES_FOR_FACES,
)
from face_recognition_tracking.embedding_extraction import EmbeddingFactory
from face_recognition_tracking.face_extraction import FaceDetectionFactory
from face_recognition_tracking.search import VectorDatabase
from face_recognition_tracking.utils import ImageHelper

embedding = (
    EmbeddingFactory.create_embedding_extractor()
)  # used to create embedding from the detected face
vectorDb = (
    VectorDatabase()
)  # used to save embedding and query similar embedding from the similar faces
face_detector = (
    FaceDetectionFactory.create_face_detector()
)  # used for face detection from the frame


def webcam_inference():
    # load the video frame
    cap = cv2.VideoCapture(0)  # using webcam for now

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        # get the faces as image from the frame
        faces = face_detector.detect_faces(frame)
        if len(faces) >= 1:
            # extract the embedding of the images
            # TODO make embedding extraction model take multiple images
            for face in faces:
                # extract the embedding of the image
                face_embedding = embedding.extract_embedding(face[FACE_FRAME])
                matched_person_name = vectorDb.match_face(face_embedding)
                ImageHelper.put_text_draw_bounding_box(
                    face[BOUNDING_BOXES_FOR_FACES], frame, matched_person_name
                )

        # track the image
        cv2.imshow("frame", frame)
        # show in the whole tracked image
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
                image = ImageHelper.load_image_from_path(image_path)
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
    image = ImageHelper.load_image_from_path(image_path)
    image_embedding = embedding.extract(image)

    # save or extract image embedding
    return vectorDb.match_face(image_embedding)
