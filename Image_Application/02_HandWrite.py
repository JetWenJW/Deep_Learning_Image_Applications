import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import numpy as np

mnist = tf.keras.datasets.mnist

# Step 1.載入 MNIST 手寫阿拉伯數字資料
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 訓練/測試資料的 X/Y 維度
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 訓練資料前10筆圖片的數字
y_train[:10]

# 顯示第1張圖片內含值
x_train[0]

# 將非0的數字轉為1，顯示第1張圖片
data = x_train[0].copy()
data[data>0]=1

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str)))
text_image

# 將非0的數字轉為1，顯示第2張圖片
data = x_train[1].copy()
data[data>0]=1

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str)))
text_image

# 顯示第1張圖片圖像

# 第一筆資料
X2 = x_train[0,:,:]

# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X2.reshape(28,28), cmap = 'gray')

# 隱藏刻度
plt.axis('off') 

# 顯示圖形
plt.show() 


# Step 2. Data Clean(此處無須進行)
# Step 3. Feature Engineering(特徵工程)
# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
# 顏色範圍：0~255，所以，公式簡化為 x / 255
# 注意，顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0
x_train_norm[0]

# Step 4.資料分割(此處無須進行，MNIST以幫忙處理完成)
# Step 5.建立模型
# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape = (28, 28)),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation = 'softmax')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

# Step 6. 模型訓練
history = model.fit(x_train_norm, y_train, epochs = 5, validation_split = 0.2)

# 檢查 history 所有鍵值
history.history.keys()

# 對訓練過程的準確率繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize = (8, 6))
plt.plot(history.history['accuracy'], 'r', label = '訓練準確率')
plt.plot(history.history['val_accuracy'], 'g', label = '驗證準確率')
plt.legend()


# 對訓練過程的損失繪圖
plt.figure(figsize = (8, 6))
plt.plot(history.history['loss'], 'r', label = '訓練損失')
plt.plot(history.history['val_loss'], 'g', label = '驗證損失')
plt.legend()

# Step 7. 評分(Score Model)
score = model.evaluate(x_test_norm, y_test, verbose = 0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')


# 實際預測 20 筆資料
# model.predict_classes 已在新版廢除
# predictions = model.predict_classes(x_test_norm)
predictions = np.argmax(model.predict(x_test_norm), axis=-1)

# 比對
print('actual    :', y_test[0:20])
print('prediction:', predictions[0:20])


# 顯示第 9 筆的機率
predictions = model.predict(x_test_norm[8:9])
print(f'0~9預測機率: {np.around(predictions[0], 2)}')

# 顯示第 9 筆圖像
X2 = x_test[8,:,:]
plt.imshow(X2.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show() 

# Step 8. 模型評估(此處暫不進行)
# Step 9. 模型部署
# 模型存檔
model.save('model.h5')

# 模型載入
model = tf.keras.models.load_model('model.h5')


# Step 10. 新資料預測
# 使用小畫家，繪製 0~9，實際測試看看
# 讀取影像並轉為單色
uploaded_file = './myDigits/0.png'
image1 = io.imread(uploaded_file, as_gray = True)

# 縮為 (28, 28) 大小的影像
image_resized = resize(image1, (28, 28), anti_aliasing=True)    
X1 = image_resized.reshape(1,28, 28) #/ 255

# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = np.abs(1-X1)

# 預測
predictions = np.argmax(model.predict(X1), axis = -1)
print(f'Predict Result{predictions}')
print("-" * 100)
# 取得模型彙總資訊
model.summary()