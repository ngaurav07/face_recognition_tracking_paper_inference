import cv2

from face_recognition_tracking.embedding_extraction.extract import EmbeddingExtractor
from face_recognition_tracking.utils import load_image_from_path

embedding = EmbeddingExtractor()


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


def image_inference(image_path: str):
    # load the image
    image = load_image_from_path(image_path)
    image_embedding = embedding.extract(image)

    # save or extract image embedding
