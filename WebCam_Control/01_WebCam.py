import cv2 as cv

# 使用正確的相機設備編號
stream = cv.VideoCapture(0)

if not stream.isOpened():
    print("No Stream:")
    exit()

fps = stream.get(cv.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))
output = cv.VideoWriter("./webcam.mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                fps = fps, frameSize = (width, height))


while True:
    ret, frame = stream.read()
    if not ret:
        print("No more Stream:")
        break
    frame = cv.resize(frame, (width, height))
    output.write(frame)
    cv.imshow("WebCam!", frame)

    cv.imshow('WebCam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv.destroyAllWindows()
