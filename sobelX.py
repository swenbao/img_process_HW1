import cv2
import numpy as np

def sobel_X(img_path):

    # load the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found")
        exit(0)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # smooth the image using GaussianBlur
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 1)

    # define a kernel for the x-axis
    kernal_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    # 因為要處理邊邊，而我們的kernel是3x3，所以要對原圖做 padding 1
    # 這樣才不會讓原圖縮小
    # padding
    padded_img = np.pad(blur_image, pad_width=1, mode='constant', constant_values=0)

    # Perform Convolution Manually
    # initialize the output image
    # 這邊我們要先將圖片的大小設定好
    # type 要設定成 float64, 因為等等要做除法, 這樣可以先確保精確度
    # 最後再轉回 uint8
    sobel_x_output = np.zeros_like(img, dtype=np.float64)

    # loop over the image to apply the Sobel filter
    for i in range(img.shape[0]): # height
        for j in range(img.shape[1]): # width
            # apply the kernel to the image
            sobel_x_output[i, j] = np.sum(padded_img[i:i+3, j:j+3] * kernal_x)
    
    # normalize the values
    # 應該有蠻多種方法可以 normalize
    # 如果不 abs 可以保留方向
    # 第一種方法 -> abs the values | normalize to 0-255 | convert to uint8
    normalized_output1 = np.abs(sobel_x_output) / np.max(sobel_x_output) * 255

    # 第二種方法，最接近投影片
    # abs the values | change outlier to bound | convert to uint8
    normalized_output2 = np.clip(np.abs(sobel_x_output), 0, 255)

    # 第三種方法（保留方向）
    # Scale Sobel output to range [0, 255] for display
    sobel_x_min = sobel_x_output.min()
    sobel_x_max = sobel_x_output.max()
    # Normalize to 0-255 range
    normalized_output3 = ((sobel_x_output - sobel_x_min) / (sobel_x_max - sobel_x_min) * 255)

    # save the output image
    cv2.imwrite("sobel_x.jpg", sobel_x_output)
    cv2.imwrite("sobel_x1.jpg", normalized_output1)
    cv2.imwrite("sobel_x2.jpg", normalized_output2)
    cv2.imwrite("sobel_x3.jpg", normalized_output3)

    return normalized_output2

sobel_X("./Dataset_OpenCvDl_Hw1/Q3_image/building.jpg")



