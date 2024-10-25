import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

x = 50  # 图片的大小

def load_single_image(img_path):
    img = load_img(img_path, color_mode='grayscale', target_size=(x, x))  # 调整图像大小为 50x50
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
    return img_array

# 加载保存的模型
model_path = 'writing_model.h5'

if os.path.exists(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    print(f'加载的模型来自 {model_path}')
    
    # 选择图片路径
    img_path = 'TEST6.jpg'  # 替换为你想预测的图片路径
    
    # 加载并预处理图片
    img_data = load_single_image(img_path)
    
    # 使用模型进行预测
    prediction = loaded_model.predict(img_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # 显示结果
    print(f'预测结果: 类别 {predicted_class}')
    
    def show_image(image_path):
        img = load_img(image_path, color_mode='grayscale', target_size=(x, x))  # 调整图像大小为 50x50
        plt.imshow(img, cmap='binary')
        plt.title(f'{predicted_class}')
        plt.axis('off')
        plt.show()
    
    # 显示图片和预测结果
    show_image(img_path)
    
else:
    print(f'模型文件 {model_path} 不存在')
