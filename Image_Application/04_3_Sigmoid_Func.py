import tensorflow as tf
import numpy as np
from skimage.transform import resize
from skimage import io


fashion_mnist = tf.keras.datasets.fashion_mnist

# Step 1.準備資料 
# 載入 MNIST 手寫阿拉伯數字資料
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

# 訓練/測試資料的 X/y 維度
print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)


# Step 2. Clean Data(此處不處理)
# Step 3. Feature Engineering
# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
# 顏色範圍：0~255，所以，公式簡化為 x / 255
# 注意，顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0
x_train_norm[0]

# Step 4. Splite Data(Train Data & Test Data)

# One-hot encoding 轉換
y_train1 = tf.keras.utils.to_categorical(y_train)
y_test1 = tf.keras.utils.to_categorical(y_test)

# Step 5. 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6. 模型訓練
history = model.fit(x_train_norm, y_train1, epochs=5, validation_split=0.2)

# Step 7. 評分(Score Model)
score = model.evaluate(x_test_norm, y_test1, verbose=0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')


print("-" * 100)
# 取得模型彙總資訊
model.summary()
