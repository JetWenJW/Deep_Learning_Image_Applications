import os
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import IPython.display
from tensorflow.python.keras import models 
import tensorflow_hub as hub

# 下載圖像
storage_url = 'https://storage.googleapis.com/download.tensorflow.org/'
content_url = storage_url + 'example_images/YellowLabradorLooking_new.jpg'
style_url = storage_url + 'example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', content_url)
style_path = tf.keras.utils.get_file('kandinsky5.jpg', style_url)

# 定義載入圖像並進行前置處理的函數
def custom_load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 定義顯示圖像的函數
def custom_imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.axis('off')
    plt.imshow(image)
    if title:
        plt.title(title)

# 載入圖像
content_image = custom_load_img(content_path)
style_image = custom_load_img(style_path)

# 繪圖
plt.subplot(1, 2, 1)
custom_imshow(content_image, '原圖')

plt.subplot(1, 2, 2)
custom_imshow(style_image, '風格圖')

# 定義還原圖像的函數
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# 自 TensorFlow Hub 下載壓縮的模型

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# 生成圖像
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)