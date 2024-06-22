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


# 用來顯示灰度圖像的直方圖（histogram）






