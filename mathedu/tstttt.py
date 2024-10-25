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

width = 32
height = 18
fixed_height, fixed_width = 3200, 1800  # 固定大小

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

# 設定數據目錄
data_dir = 'finaltest2'
x_data, y_data = load_data(data_dir)

# 檢查加載的圖像數據
print("原始圖像數據形狀:", x_data.shape)

# 處理數據
x_data = x_data.astype('float32') / 255
y_data = to_categorical(y_data - 1, num_classes=5)

# 檢查預處理後的圖像數據
print("預處理後的圖像數據形狀:", x_data.shape)

# 使用LOOCV進行訓練和評估
loo = LeaveOneOut()
accuracies = []

# 設置早停回調
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for train_index, test_index in loo.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    # 建立模型
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 訓練模型
    model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2, validation_data=(x_test, y_test))

    # 評估模型
    scores = model.evaluate(x_test, y_test, verbose=0)
    accuracies.append(scores[1])

# 顯示LOOCV結果
print('\n平均準確率:', np.mean(accuracies))

# 在整個數據集上重新訓練模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)),
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

# 在所有數據上進行預測
predictions = model.predict(x_data)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_data, axis=1)

# 將標籤範圍從0~4轉回1~5
predicted_labels += 1
true_labels += 1

# 使用 t-SNE 將數據降到 2 維，便於可視化
x_flatten = x_data.reshape(x_data.shape[0], -1)
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
x_tsne = tsne.fit_transform(x_flatten)

# 畫出預測結果的 2D 分布圖
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=predicted_labels, cmap='viridis', s=10)
plt.colorbar(scatter, label='Predicted Class')
plt.title('2D t-SNE of Image Data After Model Prediction')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# 混淆矩陣
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 顯示混淆矩陣
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(1, 6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 評估模型在所有數據上的表現
scores = model.evaluate(x_data, y_data, verbose=0)

print('\n在整個數據集上的準確率:', scores[1])

# 訓練完模型後儲存模型
model.save('modelex2_11.h5')  # 保存模型
