import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(img_path, kernel_radius=1, sigmaX=1, sigmaY=1):
    
    # load image
    img = cv2.imread(img_path)
    if img is None:
        print('Image Not Found')
        exit(0)
    
    # gaussian blur
    kernel_size = (2*kernel_radius + 1, 2*kernel_radius + 1)
    print('Kernel Size =', kernel_size)
    blur = cv2.GaussianBlur(img, kernel_size, sigmaX, sigmaY)
    
    # save blur image
    cv2.imwrite('./image1_gaussian_blur_m{}.png'.format(kernel_radius), blur)

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gaussian Filter')
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()



# for m= 1~5

# Mild Blur (small kernel, low sigma)
gaussian_filter('/Users/swenbaola/Desktop/a.png', kernel_radius=1, sigmaX=100, sigmaY=100)