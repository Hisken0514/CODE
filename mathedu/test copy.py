import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

x = 50  # 图像大小

def load_single_image(img_path):
    img = load_img(img_path, color_mode='grayscale', target_size=(x, x))  # 调整图像大小为 50x50
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 添加 batch 维度
    return img_array

# 修改路径为原始字符串格式，避免转义字符问题
img_path = r'writing\4\23.jpg'
def show_image(image_path):#+
    """#+
    显示指定路径的灰度图像并调整为 50x50 大小。#+
#+
    Parameters:#+
    image_path (str): 要显示的图像的路径。#+
#+
    Returns:#+
    None. 该函数仅用于显示图像，不返回任何值。#+
    """#+
    img = load_img(image_path, color_mode='grayscale', target_size=(x, x))  # 调整图像大小为 50x50#+
    plt.imshow(img, cmap='binary')#+
    plt.axis('off')#+
    plt.show()#+


def show_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(x, x))  # 调整图像大小为 50x50
    plt.imshow(img, cmap='binary')
    plt.axis('off')
    plt.show()

# 显示图片和预测结果
show_image(img_path)
