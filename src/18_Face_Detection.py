import cv2 as cv
import numpy as np

# 讀取圖像
img = cv.imread('../Images/group.jpg')
# 顯示原圖像
cv.imshow('Trump', img)

# 將圖像轉換為灰度圖像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 顯示灰度圖像
cv.imshow('Gray Person', gray)

# 加載Haar級聯分類器
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# 使用Haar級聯分類器檢測灰度圖像中的人臉
# scaleFactor: 每次圖像尺寸減小的比例，1.1表示每次縮小10%
# minNeighbors: 每個候選矩形需要保留的鄰近矩形數量，1表示檢測更寬鬆
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

# 打印檢測到的人臉數量
print(f'Number faces found = {len(faces_rect)}')

# 遍歷檢測到的人臉矩形區域，並在原圖像上繪製綠色矩形框
for (x, y, w, h) in faces_rect:
    # 繪製矩形框
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

# 顯示繪製了矩形框的圖像
cv.imshow('Detected Faces', img)

# 等待按鍵事件，應改為等待並檢查按下的按鍵
if cv.waitKey(0) & 0xFF == ord('d'):
    cv.destroyAllWindows()
# 無限期等待按鍵按下
cv.waitKey(0)

# 偵辨人臉時可能會失真是因為:
#     1. haar_cascade比較敏感(Because of noise)，容易導致分辨人臉時失誤
#     2. 如:眼鏡，曝光度，脖子...etc，都可能導致辨別失誤。