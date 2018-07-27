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

train_path = './imgs/train'
valid_path = './imgs/val'
class_list = ['bankcard','claim','driverlic','id','others','vehiclelic']

data_generator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=0.1,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    fill_mode='nearest')
test_generator = ImageDataGenerator(rescale=1./255)

train_batches = data_generator.flow_from_directory(train_path,target_size=(224,224),classes=class_list,batch_size=50)
valid_batches = data_generator.flow_from_directory(valid_path,target_size=(224,224),classes=class_list,batch_size=10)
# test_batches = data_generator.flow_from_directory(test_path,target_size=(224,224),classes=class_list,batch_size=1)

mobile = keras.applications.mobilenet.MobileNet()
model = Sequential()

for layer in mobile.layers:
    model.add(layer)

model.layers.pop()
model.add(Dense(6,activation='softmax'))
model.summary()

model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch=266,validation_data=valid_batches,validation_steps=50,epochs=50,verbose=2)
model.save('./moblieNet.h5')