import cv2 as cv
import numpy as np

# 讀取圖像
img = cv.imread('Images/paris.jpg')
cv.imshow('paris', img)   #cv.imshow(name, params)

# 創建一個空白畫布
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

# 創建圓形和矩形遮罩
circle = cv.circle(blank.copy(), (img.shape[1]// + 45 , img.shape[0]//2), 500, 255, -1)
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

# 使用 bitwise_and 結合圓形和矩形遮罩
weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Weird Image', weird_shape)

# 將遮罩應用到原始圖像上
masked_img = cv.bitwise_and(img, img, mask = weird_shape)
cv.imshow('Masked Image', masked_img)

# 等待按鍵事件，直到按下任意鍵時關閉所有窗口
key = cv.waitKey(0) & 0xFF
if key == ord('d'):
    cv.destroyAllWindows()
