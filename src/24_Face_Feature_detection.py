# 載入相關套件
import dlib
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from imutils import face_utils

# 載入圖檔
image_file = "/home/Jet/OpenCV/Images/two.jpg"
image = plt.imread(image_file)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.savefig('1.png')


# 載入 dlib 以 HOG 基礎的臉部偵測模型
model_file = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_file)

# 偵測圖像的臉部
rects = detector(image)

print(f'偵測到{len(rects)}張臉部.')
# 偵測每張臉的特徵點
for (i, rect) in enumerate(rects):
    # 偵測特徵點
    shape = predictor(image, rect)
    
    # 轉為 NumPy 陣列
    shape = face_utils.shape_to_np(shape)

    # 標示特徵點
    for (x, y) in shape:
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        
# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.savefig('2.png')


# 讀取視訊檔
cap = cv2.VideoCapture('./images_face/hamilton_clip.mp4')
while True:
    # 讀取一幀影像
    _, image = cap.read()
    
    # 偵測圖像的臉部
    rects = detector(image)    
    for (i, rect) in enumerate(rects):
        # 偵測特徵點
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
    
        # 標示特徵點
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # 顯示影像
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF    # 按 Esc 跳離迴圈
    if k == 27:
        break

# 關閉輸入檔    
cap.release()
# 關閉所有視窗
cv2.destroyAllWindows()