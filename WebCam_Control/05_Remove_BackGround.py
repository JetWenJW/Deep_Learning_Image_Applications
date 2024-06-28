import cv2 as cv
import numpy as np


# 使用正確的相機設備編號
stream = cv.VideoCapture("/home/Jet/OpenCV/Videos/video.mp4")

if not stream.isOpened():
    print("No Stream:")
    exit()

num_frames = stream.get(cv.CAP_PROP_FRAME_COUNT)
frame_id = np.random.uniform(size = 20) * num_frames

frames = []
for fid in frame_id:
    stream.set(cv.CAP_PROP_POS_FRAMES, fid)
    ret, frame = stream.read()
    if not ret:
        print("Something went Wrong~")
        exit()
    frames.append(frame)

median = np.median(frames, axis = 0).astype(np.uint8)
median = cv.cvtColor(median, cv.COLOR_BGR2GRAY)

fps = stream.get(cv.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))
output = cv.VideoWriter("./No_Back_Ground.mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                fps = fps, frameSize = (width, height))

stream.set(cv.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = stream.read()
    if not ret:
        print("No more Stream:")
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dif_frame = cv.absdiff(median, frame)
    threshlod, diff = cv.threshold(dif_frame, 50, 255, cv.THRESH_BINARY)


    # frame = cv.resize(frame, (width, height))
    # output.write(frame)
    cv.imshow("WebCam!", diff)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv.destroyAllWindows()
