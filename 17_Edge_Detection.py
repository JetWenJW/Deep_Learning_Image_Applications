import cv2 as cv
import numpy as np

# img = cv.imread('Images/paris.jpg')
# cv.imshow('paris', img)   #cv.imshow(name, params)
img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacing
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combine_Sobel', combined_sobel)

# canny
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)


# Sobel 算子
#   Sobel算子是一種基於梯度的算法，主要用於檢測圖像中的邊緣。
#   它分為水平方向（x方向）和垂直方向（y方向）兩個算子。
#  1. 工作原理：
#   Sobel算子通過計算圖像中每個像素點的梯度大小來檢測邊緣。
#   sobelx 和 sobely 分別計算圖像在水平和垂直方向上的梯度值。
#   通過結合這兩個方向的梯度值，可以得到邊緣的強度和方向信息。
#  2. 應用場景：
#   Sobel算子常用於需要精確檢測邊緣的應用中，例如物體偵測、圖像分割等。

# Laplacian 算子
#   Laplacian算子是一種二階微分算法，用於檢測圖像中的邊緣和圖像中的結構
#  1. 工作原理：
#   Laplacian算子計算圖像中每個像素點的二階導數，通常用來檢測圖像中的局部變化或者曲率變化。
#   它在檢測邊緣的同時也可以捕捉到圖像中的細微結構變化。
#  2. 應用場景：
#   Laplacian算子適合用於需要檢測圖像中局部變化和結構的應用，例如細節增強、形狀檢測等。

# 差異及意義:
#   Sobel主要用於檢測邊緣，它對噪聲比較敏感，但可以提供邊緣的方向信息。
#   Laplacian更加全面，可以捕捉到圖像中的結構變化，對噪聲有一定的抑制作用。
# 選擇使用：
#   如果任務主要是檢測和分割邊緣，Sobel通常是首選。
#   如果需要更全面地分析圖像的結構和細節，可以考慮使用Laplacian。




