#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os 
import numpy as np


import struct as st


# # Load the MNIST Dataset
# ## Presplit into test and train data

# In[ ]:





# In[ ]:


def read_idx(filename):
    # Taken from : https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    with open(filename, 'rb') as f:
        zero, data_type, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# In[36]:


def read_data_manual():
    #  Build data objects from local files 
    # workflow modified from : https://github.com/sadimanna/idx2numpy_array/blob/master/idx2numpyarray.py

    start_time = np.datetime64('now')

    training_images_filepath= "../data/raw/train-images-idx3-ubyte/train-images.idx3-ubyte"
    training_labels_filepath="../data/raw/train-labels-idx1-ubyte/train-labels.idx1-ubyte"

    testing_images_filepath= "../data/raw/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
    testing_labels_filepath= "../data/raw/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"


    x_train=read_idx(training_images_filepath)
    x_test=read_idx(testing_images_filepath)
    y_train=read_idx(training_labels_filepath)
    y_test=read_idx(testing_labels_filepath)

    print ("Y Train Data")
    print (y_train)
    print ("y train shape : " , y_train.shape)
    print ("x train shape : " , x_train.shape)

    end_time = np.datetime64('now')
    print ("Time of execution : %s seconds" % str(end_time-start_time))
    
    return (x_train, y_train, x_test, y_test)
    


# In[37]:


x_train, y_train, x_test, y_test= read_data_manual()


# In[31]:


def data_load_function(data_load_fx):
    if data_load_fx==str(1):
        # load data from raw files 
        print ("Loading data from raw data")
        x_train, y_train, x_test, y_test= read_data_manual()
    else:
        # Download the Dataset from AWS s3 bucket online
        # the data, split between train and test sets
        print ("Downloading data from AWS")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print ("Y Train Data")
        print (y_train)
        print ("y train shape : " , y_train.shape)
        print ("x train shape : " , x_train.shape)
    return (x_train, y_train, x_test, y_test)


# In[38]:


data_load_fx=input(""" Data Loading Options: 
To use raw files and convert to array :1 
To download from AWS S3 : 2
""")
x_train, y_train, x_test, y_test = data_load_function(data_load_fx)


# In[20]:





# In[40]:


#Peek at the Data format
# display(x_train)
# display(y_train)

# print ("__"*10)
# display(x_test)
# display(y_test)
# print(x_train.shape, y_train.shape)


# In[ ]:





# In[8]:


# Add a dimension for train and test data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[9]:


# Convert integer class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Change IMG pixel values to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#inplace division : returns a value between 0 and 1 for each pixel 
x_train /= 255
x_test /= 255

# Print Dimensions
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[10]:


# Define Batch size, # of target classes, and # of epochs
batch_size = 128
num_classes = 10
epochs = 10


# In[11]:


# Instantiate tensorflow sequential model 
## to use when there is one input and one output

model = Sequential()
# add 2d convolutional layer
## kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
## activation : Applies the rectified linear unit activation function.
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
## MaxPooling2D : max poolin operation for 2d spatial data
model.add(MaxPooling2D(pool_size=(2, 2)))
# add 2d convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten: Flattens the input - does not affect the batch size
model.add(Flatten())

# Add Densely connected NN layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#activation of last layer of classification: Interprets result as a probability distribution
model.add(Dense(num_classes, activation='softmax'))


# In[12]:


# Configure the model with losses and metrics 
##keras.losses.categorical_crossentropy =Computes the categorical crossentropy loss.
##keras.optimizers.Adadelta() = Optimizer that implements the Adadelta algorithm.
### Adadelta optimization is a stochastic gradient descent method that is based on adaptive learning rate per dimension to address two drawbacks: The continual decay of learning rates throughout training.The need for a manually selected global learning rate
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[13]:


# fit model to data - run batches by epoch
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")


# In[14]:


# evaluate model by metric
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[15]:


os.getcwd()


# In[20]:


os.chdir("../models")


# In[22]:


model.save('mnist.h5')
print("Saving the model as mnist.h5")


# In[ ]:




