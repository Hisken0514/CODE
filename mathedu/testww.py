import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns  # 用于数据可视化
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix  # 用于生成混淆矩阵

# 确保 TensorFlow 使用 GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("No GPU found, using CPU")

# 加载数据
def load_data(data_dir):
    images = []
    labels = []
    for label in range(5):  
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(50, 50))  # 调整图片大小
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# 加载和预处理数据
data_dir = 'writing'  # 替换为你的数据目录
x_data, y_data = load_data(data_dir)

print("原始图像数据形状:", x_data.shape)
x_data = x_data.astype('float32') / 255  # 数据归一化
y_data = to_categorical(y_data)          # 将标签转换为 one-hot 编码

print("预处理后的图像数据形状:", x_data.shape)

# 使用 Leave-One-Out 进行训练和评估
loo = LeaveOneOut()
accuracies = []

# 设置早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for train_index, test_index in loo.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    # 建立模型
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
    
    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2, validation_data=(x_test, y_test))
    
    # 评估模型
    scores = model.evaluate(x_test, y_test, verbose=0)
    accuracies.append(scores[1])

print('\n平均准确率:', np.mean(accuracies))

# 在整个数据集上重新训练模型
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

# 训练模型
model.fit(x_data, y_data, epochs=10, batch_size=30, verbose=2, validation_split=0.1)

# 在所有数据上进行预测
predictions = model.predict(x_data)
predicted_labels = np.argmax(predictions, axis=1)

# 使用 t-SNE 将数据降到 2 维
x_flatten = x_data.reshape(x_data.shape[0], -1)  # 将 50x50x1 展平为 2500 维
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)  # 更新为 max_iter
x_tsne = tsne.fit_transform(x_flatten)

# 绘制 t-SNE 降维后的数据分布
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=predicted_labels, cmap='viridis', s=50)
plt.colorbar(scatter, label='Predicted Class')  # 修正 colorbar 的使用
plt.title('2D t-SNE of Image Data After Model Prediction')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# 评估模型在所有数据上的表现
scores = model.evaluate(x_data, y_data, verbose=0)
print('\n在整个数据集上的准确率:', scores[1])

# 生成混淆矩阵
true_labels = np.argmax(y_data, axis=1)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
