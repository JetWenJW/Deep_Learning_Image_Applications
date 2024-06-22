import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

#Create Blank Fig
blank = np.zeros(img.shape[:2], dtype='uint8')

# GrayScale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 500, 255, -1)
masked = cv.bitwise_and(img, img, mask = mask)
cv.imshow('Mask', masked)

# Colour Histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i ,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])

plt.show()


# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)


# 直方圖是一種統計工具，用於顯示數據的分佈情況。在圖像處理中，直方圖用來表示圖像中像素強度（亮度或顏色）的分佈。
# 對於灰度圖像，直方圖顯示了每個像素值出現的次數；對於彩色圖像，則分別顯示了每個顏色通道（例如紅色、綠色、藍色）的像素分佈情況。

# 在這個程式碼中，cv.calcHist() 函數用來計算圖像的直方圖，然後使用 Matplotlib 將其視覺化出來。
# 這樣可以了解圖像中各個顏色通道的分佈情況，幫助進行後續的圖像處理和分析。