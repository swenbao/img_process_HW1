# 好的講解，第二部超讚
# https://youtu.be/LjbYKWAQA5s?si=7KLfi71UZ8fzJrZN
# https://youtu.be/7FP7ndMEfsc?si=U001Yecv8QiAbSq0
# bilateral filter 是 Spatial Gaussian * Brightness Gaussian
# 還沒有到非常理解
# 大概是 intensity 差異大的地方不要做模糊（weight 低）
# intensity 差異小的地方做模糊（weight 高）

import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilateral_filter(img_path, kernal_readius=1, sigmaColor=75, sigmaSpace=75):
    
    # load image
    img = cv2.imread(img_path)
    if img is None:
        print('Image Not Found')
        exit(0)

    # bilateral filter
    diameter = 2*kernal_readius + 1
    bilateral = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)

    # save blur image
    cv2.imwrite('./image1_bilateral_filter_m{}.png'.format(kernal_readius), bilateral)

bilateral_filter('./Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg', kernal_readius=1, sigmaColor=75, sigmaSpace=75)
bilateral_filter('./Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg', kernal_readius=5, sigmaColor=75, sigmaSpace=75)
