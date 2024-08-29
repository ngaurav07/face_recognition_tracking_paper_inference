import random
from typing import Tuple

import cv2
import numpy as np

from face_recognition_tracking.configurations import EMBEDDING_MODEL_IMAGE_SHAPE


class ImageHelper:
    _colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for j in range(10)
    ]

    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_scale = 0.6
    _font_thickness = 2
    _thickness = 3

    @staticmethod
    def load_image_from_path(image_path: str) -> np.ndarray:
        """

        Args:
            image_path:

        Returns:

        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    @staticmethod
    def resize_image(
        image: np.ndarray, size: Tuple[int, int] = EMBEDDING_MODEL_IMAGE_SHAPE
    ) -> np.ndarray:
        """

        Args:
            image:
            size:
        """
        return cv2.resize(image, size)

    @staticmethod
    def put_text_draw_bounding_box(bounding_boxes, frame, text):
        """
        Draw bounding box and draw text like person name in the frame
        Args:
            bounding_boxes: bounding boxes that needs to be drawn
            frame: image where bounding boxes and text needs to be applied
            text: text like person name in the top left of the bounding box

        """
        x, y, w, h = bounding_boxes
        x1, y1, x2, y2 = x, y, x + w, y + h
        color = ImageHelper._colors[
            int(random.randint(1, 100)) % len(ImageHelper._colors)
        ]
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        # draw bounding box
        cv2.rectangle(frame, start_point, end_point, color, ImageHelper._thickness)
        # draw text in the frame
        ImageHelper.put_text_frame(text, frame, color, start_point)

    @staticmethod
    def put_text_frame(text, frame, color, start_point):
        """
        Put text in the bounding box
        Args:
            text: text that needs to be placed in the top left of the bounding box
            frame: image where bounding boxes and text needs to be applied
            color: color that is used for the text
            start_point: start point of the bounding box

        """
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            ImageHelper._font,
            ImageHelper._font_scale,
            ImageHelper._font_thickness,
        )

        # Calculate the position to display the text
        text_x = start_point[0]  # x-coordinate of the top-left corner of the text
        text_y = start_point[1] - 10  # y-coordinate of the top-left corner of the text

        # Ensure the text is displayed above the rectangle, not off the image
        text_y = max(text_y, text_height)

        # Display the text on the image
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            ImageHelper._font,
            ImageHelper._font_scale,
            color,
            ImageHelper._font_thickness,
        )
