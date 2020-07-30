# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:00:35 2020

@author: sdjed
"""


#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
import os
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



airplane_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/airplane'
car_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/car'
cat_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/cat'
dog_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/dog'
flower_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/flower'
fruit_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/fruit'
bike_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/motorbike'
person_dt = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/person'
resized_img = 'C:/Users/sdjed/OneDrive/Desktop/exam2/data/natural_images/resized_img'

# data augmentation
# input image dimensions
img_width, img_height = 150, 150

# number of channels
img_channels = 1


list_img = os.listdir(airplane_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(airplane_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
#********************************
list_img = os.listdir(car_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(car_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
    #********************************
list_img = os.listdir(cat_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(cat_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
#********************************
list_img = os.listdir(dog_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(dog_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
    
#********************************
list_img = os.listdir(flower_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(flower_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
    
    
#********************************
list_img = os.listdir(fruit_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(fruit_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
    
   
#********************************
list_img = os.listdir(bike_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(bike_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    
#********************************
list_img = os.listdir(person_dt)
num_samples=size(list_img)
print (num_samples)

for f in list_img:
    im = Image.open(person_dt + '/' + f)  
    img = im.resize((img_width, img_height))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(resized_img +'/' +  f, "JPEG")
    
    images = os.listdir(resized_img)
    
    
    
# open one image to get the size of the images
image1 = array(Image.open(resized_img + '/'+ images[0])) 
# get the size of the images
w,h = image1.shape[0:2]
# number of images in the resized_img folder 
num_img = len(images) 
print(num_img)


# matrix of all flattened images
matrix_img = array([array(Image.open(resized_img + '/' + img2)).flatten() 
for img2 in images],'f')
               
label=np.ones((num_img,),dtype = int)
label[0:726]=0   # aireplane
label[727:1694]=1  # car
label[1695:2579]=2   #cat
label[2580:3281]=3   # dog
label[3282:4124]=4   # flower
label[4125:5124]=5   # fruit
label[5125:5912]=6  # motorbike
label[5913: ]=7 # person


data,Label = shuffle(matrix_img,label, random_state=2)
train_data = [data,Label]

img=matrix_img[170].reshape(img_width,img_height)


plt.imshow(img)
plt.show()
print (train_data[0].shape)
print (train_data[1].shape)

#**********************
#batch_size=32
#num_class=8
#nb_epochs=5
#img_width, img_height = 200, 200
#img_channels=1
#nb_filters=32
#nb_pool = 2

#nb_conv = 3

#*********************

(x,y)=(train_data[0],train_data[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train = x_train.reshape(x_train.shape[0],1,img_width,img_height)
x_test = x_test.reshape(x_test.shape[0],1,img_width,img_height)


# Store train and test data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0


print('x_train shape:', x_train.shape)
print('x_train sample:', x_train.shape[0])
print('x_test sample:', x_test.shape[0])

y_train=np_utils.to_categorical(y_train,num_class)
y_test=np_utils.to_categorical(y_test,num_class)

i=100
plt.imshow(x_train[i,0], interpolation='nearest')
print('label:',y_train[i:])



# create model

# sequential model
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,150,150), data_format='channels_first'))
model.add(Dropout(0.5))

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,150,150), data_format='channels_first'))


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(1,150,150), data_format='channels_first'))
model.add(Dropout(0.2))


#output
#  Flatten layer.
model.add(Flatten())
# Fully connected layer with 1024 units and a rectifier activation function.
model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))


# Dropout layer at 50%.
model.add(Dropout(0.5))

model.add(Dense(8, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size,epochs = epochs,
          validation_data=(x_test, y_test))

model.fit(x_train, y_train, 
          batch_size = batch_size,
          epochs = epochs,
          validation_split=0.2)
# save the model
model.save('model.h5')
# Final evaluation of the model
model = load_model('model.h5')
scores = model.evaluate(x_test, y_test, show_accuracy='True', verbose=0)
print("test Score: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))

predict= model.predict_classes(x_test[1:5])
print('predicted value:', x_test[1:5])
print('Actual value:', y_test[1:5])
