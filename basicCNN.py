import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def create_training_data(data_dir, IMG_SIZE, CLASS_NAMES):

    training_data = []
    testing_data = []
    distribution = []

    for class_name in CLASS_NAMES:
        class_count = 0
        path = os.path.join(data_dir,class_name)  # create path to flowers
        class_num = CLASS_NAMES.index(class_name)  # get the classification 0-4
        class_len = len(os.listdir(path))
        class_split = class_len * 3 // 4

        for img in os.listdir(path):  
           
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

            if class_count < class_split:
                training_data.append([new_array, class_num])

            else:
                testing_data.append([new_array, class_num])
                  # add this to our training_data or testing_data

            class_count+=1
            #plt.imshow(img_array)  # graph it
            #plt.show()
            #break
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

        distribution.append(class_count)
        
    return training_data, testing_data, distribution


def restructure_data(training_data):

    X = []
    y = []

    for features,label in training_data:
        X.append(features)
        y.append(label)
    
    X = np.array(X)

    X_shape = (-1,) + X.shape[1:]

    X.reshape(X_shape)

    return X,y


def build_model(X, y, output_size):

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
    

