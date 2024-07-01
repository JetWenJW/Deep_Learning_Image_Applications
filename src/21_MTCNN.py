# 安裝套件： pip install mtcnn
# 載入相關套件
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN

# 載入圖檔
image_file = "/home/Jet/OpenCV/Images/two.jpg"
image = plt.imread(image_file)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.savefig('1.png')

# 建立 MTCNN 物件
detector = MTCNN()

# 偵測臉部
faces = detector.detect_faces(image)

# 臉部加框
ax = plt.gca()
for result in faces:
    # 取得框的座標及寬高
    x, y, width, height = result['box']
    # 加紅色框
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)
    
    # 特徵點
    for key, value in result['keypoints'].items():
        # create and draw dot
        dot = Circle(value, radius=5, color='green')
        ax.add_patch(dot)
    
# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.savefig('2.png')

# 臉部加框
plt.figure(figsize=(8,6))
ax = plt.gca()

for i, result in enumerate(faces):
    # 取得框的座標及寬高
    x1, y1, width, height = result['box']
    x2, y2 = x1 + width, y1 + height
    
    # 顯示圖像
    plt.subplot(1, len(faces), i+1)
    plt.axis('off')
    plt.imshow(image[y1:y2, x1:x2])
plt.savefig('3.png')

















