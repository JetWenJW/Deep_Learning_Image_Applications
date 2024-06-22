import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 500, 255, -1)
cv.imshow('Circle', circle)

mask = cv.bitwise_and(gray, gray, mask = circle)
cv.imshow('Mask', mask)


# GrayScale histogram
gray_hist = cv.calcHist([gray], [0], mask, [2556], [0, 256])

plt.figure()
plt.title('GrayScale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()


# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)



# 直方圖（Histogram）的意義：
# 在影像處理中，直方圖是一種統計工具，用來顯示圖像中像素值的分佈情況。
# 對於灰度圖像而言，直方圖顯示了每個像素值（0 到 255）在圖像中出現的次數或像素數量。
# 在這個程式碼中，透過創建遮罩 circle 並將其應用到灰度圖像 gray 上，計算出了圖像中指定區域的像素值分佈，從而可以了解這個區域的亮度或灰度強度分佈情況。



