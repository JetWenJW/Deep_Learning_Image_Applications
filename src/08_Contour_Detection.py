import cv2 as cv
import numpy as np

# 讀取圖像並顯示
img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

# 創建一個與圖像大小相同的空白圖像（全黑），並顯示
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

# 將圖像轉換為灰度圖像並顯示
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 對灰度圖像進行高斯模糊，並顯示
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# 使用Canny算法進行邊緣檢測並顯示結果
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# 對灰度圖像進行二值化處理，並顯示結果
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

# 在邊緣圖像中查找輪廓
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found~')  # 打印找到的輪廓數量

# 在空白圖像上繪製找到的輪廓，並顯示
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

# 等待按鍵事件，直到按下'd'鍵時關閉所有窗口
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)
