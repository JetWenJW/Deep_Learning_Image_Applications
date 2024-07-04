import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import numpy as np
import cv2 as cv


# Step 8. 模型評估(此處暫不進行)
# Step 9. 模型部署(沿用 No.02 的 model.h5)
# 模型存檔

# Step 8. 模型評估(此處暫不進行)
# Step 9. 模型部署
# 模型存檔

# 模型載入
model = tf.keras.models.load_model('CNN_model.h5')


# Step 10. 新資料預測
# 使用小畫家，繪製 0~9，實際測試看看
# 讀取影像並轉為單色
uploaded_file = './myDigits/9.png'
image1 = io.imread(uploaded_file, as_gray = True)

# 縮為 (28, 28) 大小的影像
image_resized = resize(image1, (28, 28), anti_aliasing=True)    
X1 = image_resized.reshape(1,28, 28) #/ 255

# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = np.abs(1-X1)

# 預測
predictions = np.argmax(model.predict(X1), axis = -1)
print(f'Predict Result{predictions}')



print("-" * 40)
model.summary()