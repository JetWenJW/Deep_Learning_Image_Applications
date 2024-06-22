import cv2 as cv

img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)

# Averaging
average = cv.blur(img, (3, 3))
cv.imshow('Average', average)

# Gaussian Blur
gaussian = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('Gaussian Blur', gaussian)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral Blur
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)
# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)


# 為什麼需要對圖片進行模糊操作？
# 降噪：
#   模糊可以減少圖像中的噪聲。例如，高斯模糊通常用於平滑圖像，去除隨機噪聲。
# 圖像平滑：
#   對圖像進行平滑處理，使圖像看起來更加柔和，減少突兀的變化。
# 特徵檢測：
#   模糊有助於減少不重要的細節，使得特徵檢測（如邊緣檢測、輪廓檢測）更加穩定。例如，在應用邊緣檢測算法（如Canny邊緣檢測）之前，通常會先進行高斯模糊，以平滑圖像，減少細小邊緣的誤報。
# 圖像分割：
#   在某些圖像分割任務中，模糊處理有助於平滑不同區域之間的過渡，使分割結果更加穩定。
# 背景模糊：
#   在某些應用中（如人像攝影），模糊處理用於虛化背景，以突出主體。
# 抗鋸齒：
#   在圖像縮放或旋轉時，模糊處理可以減少鋸齒效應，使得圖像邊緣更加平滑。