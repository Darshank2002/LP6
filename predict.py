import numpy as np
from keras.utils import load_img,img_to_array
from keras.models import load_model
import train

ImagePath='D:\study\Fourth Year\Mini projects\Face Recognition\Face-Images\Face Images\Final Testing Images\darshan\Screenshot 2023-05-08 155257 - Copy - Copy - Copy - Copy.jpg'
test_image=load_img(ImagePath,target_size=(64, 64))
test_image=img_to_array(test_image)
classifier = load_model("model.keras")
 
test_image=np.expand_dims(test_image,axis=0)
 
result=classifier.predict(test_image,verbose=0)
#print(training_set.class_indices)
 
print('####'*10)
print('Prediction is: ',train.ResMap[np.argmax(result)])