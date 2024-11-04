from sobelX import sobel_X
from sobelY import sobel_Y
import cv2
import numpy as np


def gradient_angle(img_path):
    # load the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found")
        exit(0)
    
    direction_X, directionless_X = sobel_X(img_path)
    direction_Y, directionless_Y= sobel_Y(img_path)

    # combine the two images
    magnitude = np.sqrt(direction_X**2 + direction_Y**2)
    # normalize the values
    sobel_combine = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # calculate the angle
    angle = np.arctan2(direction_Y, direction_X) # -pi to pi
    degree = np.degrees(angle) + 180 # -180 to 180 -> 0 to 360

    # Create two masks based on the angle ranges using cv2.inRange
    mask1 = cv2.inRange(degree, 170, 190)
    mask2 = cv2.inRange(degree, 260, 280)

    # Perform bitwise AND with the combination image and each mask
    result1 = cv2.bitwise_and(sobel_combine, sobel_combine, mask=mask1)
    result2 = cv2.bitwise_and(sobel_combine, sobel_combine, mask=mask2)
    
    # save the output image
    cv2.imwrite("gradient_angle1.jpg", result1)
    cv2.imwrite("gradient_angle2.jpg", result2)

gradient_angle('./Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')





