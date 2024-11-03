# 好教學
# https://youtu.be/7FP7ndMEfsc?si=U001Yecv8QiAbSq0

import cv2
import numpy as np

def median_filter(img_path, kernal_readius=1):
    # load image
    img = cv2.imread(img_path)
    if img is None:
        print('Image Not Found')
        exit(0)
    
    # median filter
    # kernal 決定要取多少個 pixel 的中位數
    # 假如 kernal_readius=1, 則每一次的處理，window size 為 3x3 = 9 pixels 的正方形
    # 它會將這 9 個 pixel 的值由小到大排序，取中間的值作為這個 pixel 的值
    # 故稱之為 median filter 中位數濾波器
    median = cv2.medianBlur(img, 2*kernal_readius+1)

    # save blur image
    cv2.imwrite('./image1_median_filter_m{}.png'.format(kernal_readius), median)

median_filter('./Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg', kernal_readius=1)
median_filter('./Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg', kernal_readius=5)