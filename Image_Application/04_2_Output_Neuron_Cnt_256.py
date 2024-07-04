import tensorflow as tf
import numpy as np

# Step 1. 準備資料
# 載入 Fashion MNIST 手寫服裝資料集
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Step 2. Clean Data(此處不處理)

# Step 3. Feature Engineering
# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
# 顏色範圍：0~255，所以，公式簡化為 x / 255
# 注意，顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# Step 4. Splite Data(Train Data & Test Data)
# 此處不需要額外處理分割，因為資料已經分成訓練集和測試集

# Step 5. 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 將二維影像展平為一維
    tf.keras.layers.Dense(256, activation='relu'),  # 全連接層，256個神經元，ReLU激活函數
    tf.keras.layers.Dropout(0.2),  # Dropout層，避免過度擬合
    tf.keras.layers.Dense(10, activation='softmax')  # 輸出層，10個神經元，使用softmax激活函數
])

# Step 6. 模型訓練
model.compile(optimizer='adam',  # 優化器使用Adam
              loss='sparse_categorical_crossentropy',  # 損失函數使用稀疏分類交叉熵
              metrics=['accuracy'])  # 評估指標為準確率

history = model.fit(x_train_norm, y_train, epochs=5, validation_split=0.2)  # 訓練模型，5個epoch，20%作為驗證集

# Step 7. 評分(Score Model)
score = model.evaluate(x_test_norm, y_test, verbose=0)  # 評估模型在測試集上的表現

for i, metric in enumerate(model.metrics_names):
    print(f'{metric}: {score[i]:.4f}')  # 打印測試集上的損失值和準確率


# Step 8. 保存模型
model.save('Neuron_256.h5')
print("Model saved to Neuron_256.h5")

# Step 9. 加載模型
loaded_model = tf.keras.models.load_model('Neuron_256.h5')

# Step 10. 使用模型進行預測
predictions = loaded_model.predict(x_test_norm[:5])  # 對前5個測試樣本進行預測
predicted_classes = np.argmax(predictions, axis=1)  # 獲取預測類別

# 打印預測結果和實際類別
print("Predicted classes:", predicted_classes)
print("Actual classes:", y_test[:5])



print("-" * 100)
# 取得模型彙總資訊
model.summary()