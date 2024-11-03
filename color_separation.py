import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_separation(img_path):

    # Read the image and check if it is read correctly
    img = cv2.imread(img_path)
    if img is None:
        print('Could not open or find the image')
        exit(0)

    # Split the image into B, G, R channels
    b, g, r = cv2.split(img)

    # turn each channel back into BGR image with only one channel
    # cv2.merge() take a list of 2D arrays and merge them into a single 3D array
    # just like np.stack() but np.stack() is more general, it can stack along any axis
    # cv2.merge() is specifically for stacking along the 3rd axis
    # so generally, cv2.merge() is more efficient than np.stack() for stacking along the 3rd axis
    zeros = np.zeros_like(b)
    b_image = cv2.merge([b, zeros, zeros])
    g_image = cv2.merge([zeros, g, zeros])
    r_image = cv2.merge([zeros, zeros, r])

    # Display the images
    # figure is the whole window, figsize is the size of the window in inches
    plt.figure(figsize=(15, 5)) # 15 inches wide, 5 inches tall
    # subplot is like a specific area in the window
    # use subplot to divide the window into multiple areas
    # plt.subplot(nrows, ncols, index)
    plt.subplot(1, 3, 1)
    plt.imshow(b_image)
    plt.title('Blue channel')

    plt.subplot(1, 3, 2)
    plt.imshow(g_image)
    plt.title('Green channel')

    plt.subplot(1, 3, 3)
    plt.imshow(r_image)
    plt.title('Red channel')

    # show the window
    plt.show()

# Testing the function with a sample image
color_separation('./Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')