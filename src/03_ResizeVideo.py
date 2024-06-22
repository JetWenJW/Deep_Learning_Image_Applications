import cv2 as cv

# 讀取視頻
capture = cv.VideoCapture('Videos/video.mp4')


def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# 確認視頻文件是否成功打開
if not capture.isOpened():
    print("Error: Could not open video.")
    exit()

# 一幀一幀地讀取視頻
while True:
    isTrue, frame = capture.read()
    frame_resize = rescaleFrame(frame)


    # 如果讀取失敗，退出循環
    if not isTrue:
        print("Error: Failed to read frame or end of video.")
        break
    
    # 顯示幀
    cv.imshow('Video', frame)
    cv.imshow('Video Resize', frame_resize)

    # 等待 20 毫秒或按下 'd' 鍵退出
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# 釋放視頻捕捉對象並關閉所有 OpenCV 視窗
capture.release()
cv.destroyAllWindows()




