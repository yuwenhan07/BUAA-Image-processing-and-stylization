import cv2
import numpy as np


def sharpen_image(image):
    sharpening_filter = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(image, -1, sharpening_filter)
    return sharpened


