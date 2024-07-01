import os
import cv2 as cv
import numpy as np

people = ['Canelo', 'Donnie', 'Loma', 'Trump', 'Wife', 'JetWen']
DIR = r'/home/Jet/OpenCV/Face_Reg_IMG/'
haar_cascade = cv.CascadeClassifier('haar_face.xml')


features = []
labels = []

def create_train():
    for person in people :
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Trainning Done-------------------------')


features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_regnizer = cv.face.LBPHFaceRecognizer_create()

# Train the Regnizer on the featur list & label slist
face_regnizer.train(features, labels)

face_regnizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)





# # 等待按鍵事件，應改為等待並檢查按下的按鍵
# if cv.waitKey(0) & 0xFF == ord('d'):
#     cv.destroyAllWindows()
# # 無限期等待按鍵按下
# cv.waitKey(0)