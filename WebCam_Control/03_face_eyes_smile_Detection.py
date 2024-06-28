import cv2 as cv


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

def detect_features(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y , w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), color = (0, 0, 255), thickness = 5)
        face = frame[y : y + h, x : x + w]
        gray_face = gray[y : y + h, x : x + w]

        smiles = smile_cascade.detectMultiScale(gray_face, 2.5, minNeighbors = 9)
        for(xp, yp, wp, hp) in smiles:
            face = cv.rectangle(face, (xp, yp), (xp + wp, yp + hp), color = (0, 255, 0), thickness = 5)
        
        eyes = eye_cascade.detectMultiScale(gray_face, 2.5, minNeighbors = 7)
        for(xp, yp, wp, hp) in eyes:
            face = cv.rectangle(face, (xp, yp), (xp + wp, yp + hp), color = (255, 0, 0), thickness = 5)
    
    
    return frame

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
    
    #output.write(frame)
    cv.imshow("WebCam!", frame)

    frame = detect_features(frame)

    cv.imshow('WebCam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv.destroyAllWindows()
