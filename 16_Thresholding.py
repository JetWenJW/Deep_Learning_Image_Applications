import cv2 as cv

img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded_INV', thresh_inv)

# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)


# 為什麼需要Thresholding？
#   1. 分割圖像：Thresholding可以將圖像中的像素分為兩類或多類，根據像素值是否超過或達到一個特定的閾值。
#   2. 提取感興趣的區域：在許多情況下，我們希望提取圖像中特定亮度或顏色的區域，Thresholding能夠幫助我們實現這一目標。
#   3. 去除噪點：在一些應用中，Thresholding也可以用來去除圖像中的噪點，只保留我們感興趣的主要特徵。

# Simple Thresholding（簡單閾值化）
#   -->Simple Thresholding是一種最基本的Thresholding方法，基於全域閾值對整個圖像進行處理。其運作原理如下：
#   1. 設定閾值：通過設定一個固定的閾值（例如150），將圖像中每個像素的灰度值與這個閾值進行比較。
#   2. 分類像素：如果像素的灰度值大於閾值，則將其設置為一個值（通常是255，白色），否則設置為另一個值（通常是0，黑色）。
#   3. 應用場景：適用於圖像中對比明顯且背景與前景區域分明的情況。

# Adaptive Thresholding（自適應閾值化）
#   -->Adaptive Thresholding則是一種更靈活的Thresholding方法，它允許根據圖像局部區域的特性來調整閾值，適應不同的光照條件和區域內的變化。
#     其特點包括：
#   1. 局部閾值計算：不像Simple Thresholding使用全域閾值，Adaptive Thresholding會根據圖像的每個小區域來計算不同的閾值。
#   2. 處理變化光照：適應性閾值可以有效應對圖像中光照不均勻或變化較大的情況。
#   3. 應用場景：適用於光照變化大的圖像，如影像中存在陰影或光照不均的情況。






