# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:34:47 2018

@author: soumyama
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


tf.logging.set_verbosity(tf.logging.INFO)
sess=tf.InteractiveSession()

#Import Train and Test data

train=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\train.csv")
print("Shape of the Train data with all the features: ", train.shape)
train= train.select_dtypes(exclude=['object'])
print("")
print("Shape of the train data with only numerical features: ", train.shape )
train.drop('Id', axis=1, inplace=True)
train.fillna(0, inplace=True)


test=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\test.csv")
print("Shape of the Test data with all the features: ", test.shape)
test= test.select_dtypes(exclude=['object'])
print("")
print("Shape of the test data with only numerical features: ", test.shape )
test.drop('Id', axis=1, inplace=True)
test.fillna(0, inplace=True)

print("")
print("List of features contained in our dataset", list(train.columns))


#Outliers

from sklearn.ensemble import IsolationForest

clf=IsolationForest(max_samples=100, random_state=42)
clf.fit(train)

y_noano= clf.predict(train)
y_noano= pd.DataFrame(y_noano, columns=['Top'])
y_noano[y_noano['Top']==1].index.values


train=train.iloc[y_noano[y_noano['Top']==1].index.values]
train.reset_index(drop=True, inplace=True)
print("number of outliers: ", y_noano[y_noano['Top']==-1].shape[0])
print("Numbero of rows without outliers: ", train.shape[0])


train.head(10)


#Preprocessing

import warnings
warnings.filterwarnings('ignore')

col_train=list(train.columns)
col_train_bis=list(train.columns)

col_train_bis.remove('SalePrice')
col_train_bis

mat_train=np.matrix(train)
mat_test=np.matrix(test)
mat_new=np.matrix(train.drop('SalePrice', axis=1))
mat_y=np.array(train.SalePrice).reshape((1314,1))

mat_test.shape

prepro_y=MinMaxScaler()
prepro_y.fit(mat_y)

prepro=MinMaxScaler()
prepro.fit(mat_train)

prepro_test=MinMaxScaler()
prepro_test.fit(mat_new)

train=pd.DataFrame(prepro.transform(mat_train), columns=col_train)
test=pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_bis)

train.head()


#List of Features
COLUMNS= col_train
FEATURES=col_train_bis
LABEL="SalePrice"

#Columns for Tensorflow
feature_cols=[tf.contrib.layers.real_valued_column(k) for k in FEATURES]

#Training set and Prediction set with the features to predict
training_set=train[COLUMNS]
prediction_set=train.SalePrice

#Train and Test
x_train, x_test, y_train, y_test=train_test_split(training_set[FEATURES], prediction_set, 
                                                  test_size=0.33, random_state=42)

y_train=pd.DataFrame(y_train, columns=[LABEL])

training_set=pd.DataFrame(x_train, columns=FEATURES).merge(y_train, left_index=True, right_index=True)
training_set.head()

#Training for submission
training_sub=training_set[col_train]


#Same thing for test set
y_test=pd.DataFrame(y_test, columns=[LABEL])
testing_set=pd.DataFrame(y_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)
testing_set.head()










