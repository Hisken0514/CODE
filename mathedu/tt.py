import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 確保 TensorFlow 使用 GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("No GPU found, using CPU")

def load_data(data_dir):
    images = []
    labels = []
    for label in range(5):  
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(50, 50))  # 調整圖片大小
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

data_dir = 'writing'  # 替換為你的數據目錄
x_data, y_data = load_data(data_dir)

# 檢查加載的圖像數據
print("原始圖像數據形狀:", x_data.shape)

# 處理數據
x_data = x_data.astype('float32') / 255
y_data = to_categorical(y_data)

# 檢查預處理後的圖像數據
print("預處理後的圖像數據形狀:", x_data.shape)

# 在整個數據集上重新訓練模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(x_data, y_data, epochs=10, batch_size=30, verbose=2, validation_split=0.1)

# 評估模型在所有數據上的表現
scores = model.evaluate(x_data, y_data, verbose=0)
print('\n在整個數據集上的準確率:', scores[1])

# 保存模型
model.save('writing_model.h5')
