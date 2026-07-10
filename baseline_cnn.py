# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 10:26:45 2022

@author: Jimmy

TensorFlow/Keras reference CNN -- matches the architecture described in the
project report (single Conv2D(32, 3x3)+BN+ReLU+MaxPool block, four Dense(256,
ReLU) layers, softmax output; Adam, 8 epochs, batch size 16). Included here
as the reference/baseline implementation; the actively validated pipeline in
this repo is the PyTorch one (src/pfdataset.py + src/motor_fault_cnn.py).

Unmodified from the original project script.
"""

# import matplotlib
# matplotlib.use('QT5Agg')
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth = True))
sess = tf.compat.v1.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES']='0'

import scipy.io as scio
import pandas as pd

def data_load(filepath,labelname):
    data = scio.loadmat(filepath)
    data = data['data']
    data = pd.DataFrame(data, \
                        columns=['Voltage_0','Voltage_1','Voltage_2','Current_0',\
                                 'Current_1','Current_2','Rotor_Current','Speed',\
                                     'Failde',labelname])
    return data

filepath = 'Preprocessed_Disconnect_Phase_10_11_21_.mat'
labelname = 'Disconnect_Phase_10_11_21_'
data_Preprocessed_Disconnect_Phase_10_11_21 = data_load(filepath,labelname)

filepath = 'Preprocessed_No_failed.mat'
labelname = 'No_failed'
data_Preprocessed_No_failed = data_load(filepath,labelname)

filepath = 'Preprocessed_Rotor_Current_Failed_R_.mat'
labelname = 'Rotor_Current_Failed_R_'
data_Preprocessed_Rotor_Current_Failed_R = data_load(filepath,labelname)

filepath = 'Preprocessed_Short_between_two_phases_.mat'
labelname = 'Short_between_two_phases_'
data_Preprocessed_Short_between_two_phases = data_load(filepath,labelname)

filepath = 'Preprocessed_Test_Data_Rotor_Current_Faild.mat'
labelname = 'Test_Data_Rotor_Current_Faild'
data_Preprocessed_Test_Data_Rotor_Current_Faild = data_load(filepath,labelname)

filepath = 'Preprocessed_Test_Data_Short_phases_Ln_G_.mat'
labelname = 'Test_Data_Short_phases_Ln_G_'
data_Preprocessed_Test_Data_Short_phases_Ln_G = data_load(filepath,labelname)
   


# dataframe data/  array label
def data_processing(data):
    data = data.iloc[:,0:8]
    row = data.shape[0]
    data = np.array(data)
    data = data.reshape(int(row/10000),10000,8)
    return data

def downsample(data,sample_time=4):
    row = int(data.shape[0])
    data1 = np.zeros(shape=(int(data.shape[0]*sample_time),int(data.shape[1]/sample_time),data.shape[2]))
    x1 = np.arange(0,int(data.shape[1]),sample_time)
    x2 = np.arange(1,int(data.shape[1]),sample_time)
    x3 = np.arange(2,int(data.shape[1]),sample_time)
    x4 = np.arange(3,int(data.shape[1]),sample_time)
    for i in range(0,row):
        data1[sample_time*i:sample_time*i+1,:,:] = data[i:i+1,x1,:]
        data1[sample_time*i+1:sample_time*i+2,:,:] = data[i:i+1,x2,:]
        data1[sample_time*i+2:sample_time*i+3,:,:] = data[i:i+1,x3,:]
        data1[sample_time*i+3:sample_time*i+4,:,:] = data[i:i+1,x4,:]
    return data1


'''
label 0  / Test_Data_Rotor_Current_Faild
''' 
data_Preprocessed_Test_Data_Rotor_Current_Faild = data_processing(data_Preprocessed_Test_Data_Rotor_Current_Faild)
data_Preprocessed_Test_Data_Rotor_Current_Faild = downsample(data_Preprocessed_Test_Data_Rotor_Current_Faild)
row0 = data_Preprocessed_Test_Data_Rotor_Current_Faild.shape[0]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild = 0*np.ones(shape=(int(row0),1))

'''
label 1  / Disconnect_Phase_10_11_21
'''

data_Preprocessed_Disconnect_Phase_10_11_21 = data_processing(data_Preprocessed_Disconnect_Phase_10_11_21)
data_Preprocessed_Disconnect_Phase_10_11_21 = downsample(data_Preprocessed_Disconnect_Phase_10_11_21)
row1 = data_Preprocessed_Disconnect_Phase_10_11_21.shape[0]
label_data_Preprocessed_Disconnect_Phase_10_11_21 = 1*np.ones(shape=(int(row1),1))


'''
label 2 / Rotor_Current_Failed_R
'''
data_Preprocessed_Rotor_Current_Failed_R = data_processing(data_Preprocessed_Rotor_Current_Failed_R)
data_Preprocessed_Rotor_Current_Failed_R = downsample(data_Preprocessed_Rotor_Current_Failed_R)
row2 = data_Preprocessed_Rotor_Current_Failed_R.shape[0]
label_data_Preprocessed_Rotor_Current_Failed_R = 2*np.ones(shape=(int(row2),1))

'''
label 3 / Short_between_two_phases
'''
data_Preprocessed_Short_between_two_phases = data_processing(data_Preprocessed_Short_between_two_phases)
data_Preprocessed_Short_between_two_phases = downsample(data_Preprocessed_Short_between_two_phases)
row3 = data_Preprocessed_Short_between_two_phases.shape[0]
label_data_Preprocessed_Short_between_two_phases = 3*np.ones(shape=(int(row3),1))

'''
label 4 / Test_Data_Short_phases_Ln_G
'''
data_Preprocessed_Test_Data_Short_phases_Ln_G = data_processing(data_Preprocessed_Test_Data_Short_phases_Ln_G)
data_Preprocessed_Test_Data_Short_phases_Ln_G = downsample(data_Preprocessed_Test_Data_Short_phases_Ln_G)
row4 = data_Preprocessed_Test_Data_Short_phases_Ln_G.shape[0]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G = 4*np.ones(shape=(int(row4),1))

'''
label 5 / No_failed
''' 
data_Preprocessed_No_failed = data_processing(data_Preprocessed_No_failed)
data_Preprocessed_No_failed = downsample(data_Preprocessed_No_failed)
row5 = data_Preprocessed_No_failed.shape[0]
label_data_Preprocessed_No_failed = 5*np.ones(shape=(int(row5), 1))




# pick up the minimum row / rebulid data
row_for_all = int(np.min([row0, row1, row2, row3, row4, row5]))-int(0.1*np.min([row0, row1, row2, row3, row4, row5]))
data_Preprocessed_Test_Data_Rotor_Current_Faild_ = data_Preprocessed_Test_Data_Rotor_Current_Faild[0:row_for_all,:,:]
data_Preprocessed_Disconnect_Phase_10_11_21_ = data_Preprocessed_Disconnect_Phase_10_11_21[0:row_for_all,:,:]
data_Preprocessed_Rotor_Current_Failed_R_ = data_Preprocessed_Rotor_Current_Failed_R[0:row_for_all,:,:]
data_Preprocessed_Short_between_two_phases_ = data_Preprocessed_Short_between_two_phases[0:row_for_all,:,:]
data_Preprocessed_Test_Data_Short_phases_Ln_G_ = data_Preprocessed_Test_Data_Short_phases_Ln_G[0:row_for_all,:,:]
data_Preprocessed_No_failed_ = data_Preprocessed_No_failed[0:row_for_all,:,:]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild_ = label_data_Preprocessed_Test_Data_Rotor_Current_Faild[0:row_for_all,:]
label_data_Preprocessed_Disconnect_Phase_10_11_21_ = label_data_Preprocessed_Disconnect_Phase_10_11_21[0:row_for_all,:]
label_data_Preprocessed_Rotor_Current_Failed_R_ = label_data_Preprocessed_Rotor_Current_Failed_R[0:row_for_all,:]
label_data_Preprocessed_Short_between_two_phases_ = label_data_Preprocessed_Short_between_two_phases[0:row_for_all,:]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G_ = label_data_Preprocessed_Test_Data_Short_phases_Ln_G[0:row_for_all,:]
label_data_Preprocessed_No_failed_ = label_data_Preprocessed_No_failed[0:row_for_all,:]

data_for_train = np.concatenate([data_Preprocessed_Test_Data_Rotor_Current_Faild_,data_Preprocessed_Disconnect_Phase_10_11_21_,\
                               data_Preprocessed_Rotor_Current_Failed_R_,data_Preprocessed_Short_between_two_phases_,\
                                   data_Preprocessed_Test_Data_Short_phases_Ln_G_,data_Preprocessed_No_failed_],axis=0)
label_for_train = np.concatenate([label_data_Preprocessed_Test_Data_Rotor_Current_Faild_,label_data_Preprocessed_Disconnect_Phase_10_11_21_,\
                                label_data_Preprocessed_Rotor_Current_Failed_R_,label_data_Preprocessed_Short_between_two_phases_,\
                                    label_data_Preprocessed_Test_Data_Short_phases_Ln_G_,label_data_Preprocessed_No_failed_],axis=0)

    
 # for test
data_Preprocessed_Test_Data_Rotor_Current_Faild_ = data_Preprocessed_Test_Data_Rotor_Current_Faild[row_for_all:row0+1,:,:]
data_Preprocessed_Disconnect_Phase_10_11_21_ = data_Preprocessed_Disconnect_Phase_10_11_21[row_for_all:row1+1,:,:]
data_Preprocessed_Rotor_Current_Failed_R_ = data_Preprocessed_Rotor_Current_Failed_R[row_for_all:row2+1,:,:]
data_Preprocessed_Short_between_two_phases_ = data_Preprocessed_Short_between_two_phases[row_for_all:row3+1,:,:]
data_Preprocessed_Test_Data_Short_phases_Ln_G_ = data_Preprocessed_Test_Data_Short_phases_Ln_G[row_for_all:row4+1,:,:]
data_Preprocessed_No_failed_ = data_Preprocessed_No_failed[row_for_all:row5+1,:,:]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild_ = label_data_Preprocessed_Test_Data_Rotor_Current_Faild[row_for_all:row0+1,:]
label_data_Preprocessed_Disconnect_Phase_10_11_21_ = label_data_Preprocessed_Disconnect_Phase_10_11_21[row_for_all:row1+1,:]
label_data_Preprocessed_Rotor_Current_Failed_R_ = label_data_Preprocessed_Rotor_Current_Failed_R[row_for_all:row2+1,:]
label_data_Preprocessed_Short_between_two_phases_ = label_data_Preprocessed_Short_between_two_phases[row_for_all:row3+1,:]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G_ = label_data_Preprocessed_Test_Data_Short_phases_Ln_G[row_for_all:row4+1,:]
label_data_Preprocessed_No_failed_ = label_data_Preprocessed_No_failed[row_for_all:row5+1,:]

data_for_test = np.concatenate([data_Preprocessed_Test_Data_Rotor_Current_Faild_,data_Preprocessed_Disconnect_Phase_10_11_21_,\
                               data_Preprocessed_Rotor_Current_Failed_R_,data_Preprocessed_Short_between_two_phases_,\
                                   data_Preprocessed_Test_Data_Short_phases_Ln_G_,data_Preprocessed_No_failed_],axis=0)
label_for_test = np.concatenate([label_data_Preprocessed_Test_Data_Rotor_Current_Faild_,label_data_Preprocessed_Disconnect_Phase_10_11_21_,\
                                label_data_Preprocessed_Rotor_Current_Failed_R_,label_data_Preprocessed_Short_between_two_phases_,\
                                    label_data_Preprocessed_Test_Data_Short_phases_Ln_G_,label_data_Preprocessed_No_failed_],axis=0)     
    
#Nomalizaiton for x_train
row = data_for_train.shape[0]
for i in range (row):
    data_for_train[i:i+1,:,:] = \
        (data_for_train[i:i+1,:,:]-np.min(data_for_train[i:i+1,:,:],axis=1))/\
            (np.max(data_for_train[i:i+1,:,:],axis=1) - np.min(data_for_train[i:i+1,:,:], axis=1))
data_for_train = data_for_train.reshape(data_for_train.shape[0],data_for_train.shape[1],data_for_train.shape[2],1)

num = np.random.permutation(data_for_train.shape[0])
data_for_train = data_for_train[num,:]
label_for_train = label_for_train[num,:] 

row = data_for_test.shape[0]
for i in range (row):
    data_for_test[i:i+1,:,:] = \
        (data_for_test[i:i+1,:,:]-np.min(data_for_test[i:i+1,:,:],axis=1))/\
            (np.max(data_for_test[i:i+1,:,:],axis=1) - np.min(data_for_test[i:i+1,:,:], axis=1))
data_for_test = data_for_test.reshape(data_for_test.shape[0],data_for_test.shape[1],data_for_test.shape[2],1)

###############################################    model   ###############################################

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        
        self.c1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = Dense(256, activation='relu')
        self.f2 = Dense(256, activation='relu')
        self.f3 = Dense(256, activation='relu')
        self.f4 = Dense(256, activation='relu')
        self.f5 = Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        y = self.f5(x)
        return y

model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(data_for_train, label_for_train, batch_size=16, epochs=8,validation_split=0.1)
model.summary()

y_pre = model.predict(data_for_test)
y_pre = np.argmax(y_pre,axis=1).reshape(-1,1)
error = label_for_test-y_pre
num = np.count_nonzero(error)
accuracy = (1-num/label_for_test.shape[0])*100
print('Predictin Accuracy%',accuracy)

from sklearn.metrics import confusion_matrix 
matrix = confusion_matrix(label_for_test,y_pre)
print(matrix)
###############################################    show   ###############################################

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
