import numpy as np
from keras.models import load_model
import cv2
faceCascade = cv2.CascadeClassifier('D:\study\Fourth Year\Mini projects\lp5\haarcascade_frontalface_default.xml')
import train

model = load_model('model.keras')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = img
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(64,64))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(-1,64,64,3))
        result=model.predict(reshaped)
        print(train.ResMap[np.argmax(result)])
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()