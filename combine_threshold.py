from sobelX import sobel_X
from sobelY import sobel_Y
import cv2
import numpy as np

def combine_threshold(img_path):
    
    X = sobel_X(img_path)
    Y = sobel_Y(img_path)

    # combine the two images
    magnitude = np.sqrt(X**2 + Y**2)

    # normalize the values
    normalized_output = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # threshold the image
    # threshold = 128
    _, result = cv2.threshold(normalized_output, 128, 255, cv2.THRESH_BINARY)
    # threshold = 28
    _, result2 = cv2.threshold(normalized_output, 28, 255, cv2.THRESH_BINARY)

    # save the output image
    cv2.imwrite("combined.jpg", normalized_output)
    cv2.imwrite("128threshold.jpg", result)
    cv2.imwrite("28threshold.jpg", result2)

combine_threshold('Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')

