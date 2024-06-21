import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Images/paris.jpg')
cv.imshow('paris', img)   #cv.imshow(name, params)


# BGR to Graysacle
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# HSV --> BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR', hsv_bgr)

# LAB --> BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', lab_bgr)

# 等待按鍵事件，直到按下'd'鍵時關閉所有窗口
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)




# 為什麼使用HSV ?
#   1. 人類視覺的相似性：HSV顏色空間更接近人類感知顏色的方式。調整HSV中的參數通常比調整RGB中的參數更直觀。
#   2. 顏色分割和識別：在圖像處理應用中，例如顏色分割和目標識別，使用HSV顏色空間往往更容易。
#      例如，在HSV空間中，顏色分割可以通過僅調整色調來實現，而不需要考慮亮度和飽和度的變化。
#   3. 色調獨立性：HSV空間將顏色（色調）與亮度分離，這使得在不同光照條件下處理圖像更方便。

# 為什麼使用 LAB ?
#   1. 亮度與色度分離：在LAB顏色空間中，亮度與顏色信息分離，這使得在調整顏色時不會影響亮度，反之亦然。
#   2. 顏色差異計算：LAB顏色空間的設計使得它適合計算顏色差異，例如用於顏色匹配和顏色分割。
#   3. 色彩校正：在圖像處理中，LAB顏色空間被廣泛應用於色彩校正和顏色增強，因為它能更精確地表示人眼感知的顏色。