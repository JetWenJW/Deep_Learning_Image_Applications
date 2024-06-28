import cv2 as cv

# 檢查前 10 個相機設備編號
for i in range(10):
    stream = cv.VideoCapture(i)
    if stream.isOpened():
        print(f"Camera {i} is available.")
        stream.release()
    else:
        print(f"Camera {i} is not available.")

print("Finished checking cameras.")
