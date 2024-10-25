import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import cv2

width = 64
height = 36
fixed_height, fixed_width = 1600,900   # 固定大小

# 定義填充和調整大小的函數
def pad_and_resize_image(image, target_size, fixed_size):
    img_height, img_width = image.shape[:2]
    pad_vertical = max((fixed_size[0] - img_height) // 2, 0)
    pad_horizontal = max((fixed_size[1] - img_width) // 2, 0)

    # 使用白色填充
    padded_image = cv2.copyMakeBorder(image, 
                                      top=pad_vertical, 
                                      bottom=pad_vertical, 
                                      left=pad_horizontal, 
                                      right=pad_horizontal, 
                                      borderType=cv2.BORDER_CONSTANT, 
                                      value=0)

    # 調整大小
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image

# 加載數據的函數
def load_data(data_dir):
    images = []
    labels = []
    for label in range(1, 6):
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale')
            img_array = img_to_array(img).astype('uint8')  # 轉換為整數類型
            img_resized = pad_and_resize_image(img_array, (width, height), (fixed_height, fixed_width))
            images.append(img_resized)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# 加載圖像並進行處理
img_path = 'finaltest2/1/01_0151110.jpg'  # 替換為你實際的圖片路徑
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 加載灰度圖像

# 確認圖像已成功加載
if img is not None:
    resized_img = pad_and_resize_image(img, (width, height), (fixed_height, fixed_width))

    # 顯示處理後的圖片
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)  # 等待按鍵
    cv2.destroyAllWindows()  # 關閉視窗
else:
    print("圖像加載失敗。請檢查文件路徑是否正確。")



