
from glob import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import cv2


mobile = keras.applications.mobilenet.MobileNet(weights=None)
model = Sequential()

for layer in mobile.layers:
    model.add(layer)

model.layers.pop()
model.add(Dense(2,activation='softmax'))

model.load_weights('./mobileNet_invoice.h5')
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

def load_image(img_arr):
    img_tensor = cv2.resize(img_arr,(224,224))
    img_tensor = np.expand_dims(img_tensor,axis=0)
    rgb = img_tensor[...,::-1]*1.0/255
    return rgb

def is_invoice(img_arr):
    img_tensor = load_image(img_arr)
    pred = model.predict(img_tensor)
    if np.argmax(pred[0])==0:
        print('invoice')
        return True
    else:
        print('others')
        return False

if __name__=='__main__':
    img_dir = './invoice/*.jpg'
    for img_path in glob(img_dir):
        img_arr=cv2.imread(img_path)
        is_invoice(img_arr)
