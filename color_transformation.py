import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_transformation(img_path):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        exit(0)
    
    # Convert the image to grayscale using cv2.cvtColor()
    cv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image by averaging the RGB channels
    b, g, r = cv2.split(img)
    avg_gray = (b/3 + g/3 + r/3).astype(np.uint8)

    # display the images
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv_gray, cmap='gray')
    plt.title('CV Grayscale')

    plt.subplot(1, 2, 2)
    plt.imshow(avg_gray, cmap='gray')
    plt.title('Average Grayscale')

    plt.show()

# Testing the function with a sample image
color_transformation('./Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')
