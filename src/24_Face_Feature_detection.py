import dlib
import cv2
from imutils import face_utils
import matplotlib.pyplot as plt

# 載入圖像檔案
image_file = "./images_face/jared_1.jpg"
image = cv2.imread(image_file)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換顏色通道為 RGB

# 顯示圖像
plt.imshow(image_rgb)
plt.axis('off')


# 載入 dlib 的人臉偵測模型和特徵點預測模型
model_file = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_file)

# 偵測圖像中的人臉
rects = detector(image_rgb)

print(f'偵測到{len(rects)}張臉.')

# 遍歷每張偵測到的人臉
for (i, rect) in enumerate(rects):
    # 偵測特徵點
    shape = predictor(image_rgb, rect)
    shape = face_utils.shape_to_np(shape)  # 轉為 NumPy 陣列

    # 標記特徵點
    for (x, y) in shape:
        cv2.circle(image_rgb, (x, y), 10, (0, 255, 0), -1)

# 顯示帶有特徵點的圖像
plt.imshow(image_rgb)
plt.axis('off')
plt.savefig('feature_Point.png')
print("-" * 10, "特徵點標記完成", "-" * 10)


# 讀取視訊檔案
cap = cv2.VideoCapture('./images_face/short_hamilton_clip.mp4')

while True:
    # 讀取一幀圖像
    ret, image = cap.read()
    if not ret:
        break
    
    # 轉換為灰度圖像，用於人臉偵測
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 偵測圖像中的人臉
    rects = detector(gray)
    
    # 遍歷每張偵測到的人臉
    for (i, rect) in enumerate(rects):
        # 偵測特徵點
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # 轉為 NumPy 陣列
    
        # 標記特徵點
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # 顯示圖像
    cv2.imshow("Output", image)

    # 按 Esc 鍵退出迴圈
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# 關閉視訊檔案
cap.release()
# 關閉所有視窗
cv2.destroyAllWindows()
