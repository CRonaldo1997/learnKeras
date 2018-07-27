import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


test_generator = ImageDataGenerator(rescale=1./255)
test_img_path = './test_img'
class_list = ['bankcard','claim','driverlic','id','others','vehiclelic']
test_batches = test_generator.flow_from_directory(test_img_path,target_size=(224,224),classes=class_list,batch_size=1)

mobile = keras.applications.mobilenet.MobileNet(weights=None)
model = Sequential()

for layer in mobile.layers:
    model.add(layer)

model.layers.pop()
model.add(Dense(6,activation='softmax'))

model.load_weights('./mobileNet.h5')
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

prediction = model.predict_generator(test_batches,steps=1,verbose=1)
print(prediction)