import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_extraction(img_path):
    
    # load image
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        exit(0)
    
    # convert image to HSV
    # Hue 有點像各波長顏色，就是紅橙紅綠藍靛紫，學名叫色調 (0~180)
    # saturation 就是飽和度 (0~255)
    # value 就是亮度 (0~255)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    # cv2.inRange() 根據你的設定的 upper/lower bound，將符合範圍的像素設為255，不符合的設為0
    # 所以會得到一張黑白圖片，圖片大小跟你送進去的圖片一樣，只是裡面的像素值只有 0 或 255
    # 255 的 2 進位是 11111111，0 的 2 進位是 00000000
    # 所以只要去 AND 圖片，就可以留下符合範圍的像素
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # invert the mask for later removal
    # 這一題是要去除指定範圍內的顏色，所以要取反
    mask_inverse = cv2.bitwise_not(mask)

    # apply the mask
    extracted_image = cv2.bitwise_and(img, img, mask=mask_inverse)
    
    # show the image
    plt.figure(figsize=(10,10))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Extracted Image')
    plt.imshow(cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB))

    plt.show()

color_extraction('./Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')
