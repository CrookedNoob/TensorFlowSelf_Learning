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
ID=test.Id
test.fillna(0, inplace=True)
test.drop('Id', axis=1, inplace=True)
print("")
print("Shape of the test data with only numerical features: ", test.shape )


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
test.shape

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
y_test.shape
testing_set=pd.DataFrame(x_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)
testing_set.head()
testing_set.shape

#Deep Neural Network for continuous Features

#Model
tf.logging.set_verbosity(tf.logging.ERROR)
regressor=tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                        activation_fn=tf.nn.relu, hidden_units=[200, 100, 50,25,12])

#Reset the index for training
training_set.reset_index(drop=True, inplace=True)

def input_fn(data_set, pred=False):
    if pred==False:
        feature_cols={k: tf.constant(data_set[k].values) for k in FEATURES}
        labels=tf.constant(data_set[LABEL].values)
        
        return feature_cols, labels
    
    if pred==True:
        feature_cols={k:tf.constant(data_set[k].values) for k in FEATURES}
        
        return feature_cols
    
    

#Deep Neural Regressor which contains the data split by Train and Test split
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)


#Eavluation on the test set created by the train_test_split
ev=regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

#Display the score on the testing set
loss_score1=ev["loss"]
print("Final loss on the tesing set: {0:f}".format(loss_score1))

#Predictions
y=regressor.predict(input_fn=lambda: input_fn(testing_set))
predictions=list(itertools.islice(y, testing_set.shape[0]))

testing_set.shape


#Prediction vs Submission
predictions=pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(434,1)), 
                         columns=['Prediction'])

predictions.head()
predictions.shape


reality=pd.DataFrame(prepro.inverse_transform(testing_set), columns=[COLUMNS]).SalePrice

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

fig,ax=plt.subplots(figsize=(5,5))

plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel("Predictions", fontsize=10)
plt.xlabel("Reality", fontsize=10)
plt.title('Predictions X Reality on Dataset Test', fontsize=10)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()


y_predict=regressor.predict(input_fn=lambda: input_fn(test, pred=True))


def to_submit(pred_y, name_out):
    y_predict=list(itertools.islice(pred_y, test.shape[0]))
    y_predict=pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns=['SalePrice'])
    y_predict=y_predict.join(ID)
    y_predict.to_csv(name_out+'.csv', index=False)

to_submit(y_predict, "submission_continuous")
    


#LEAKY RELU
def leaky_relu(x):
    return tf.nn.relu(x)-0.01*tf.nn.relu(-x)


#Model
regressor=tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                        activation_fn=leaky_relu,
                                        hidden_units=[200, 100, 50, 25,12])

#Deep neural network Regressor with tarining set which contains the data split by train and test split

regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000) 


#Evaluation on the test set created by the train test split
ex=regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

#Display the score on the etsting set
loss_score2=ev["loss"]
print("final loss on the testing set with Leaky RELU:  {0:f}".format(loss_score2))



#Predictions
y_predict = regressor.predict(input_fn=lambda: input_fn(test, pred = True))
to_submit(y_predict, "Leaky_relu")


# Model
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.elu, hidden_units=[200, 100, 50, 25, 12])
    
# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)

# Evaluation on the test set created by train_test_split
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

loss_score3 = ev["loss"]
print("Final Loss on the testing set with Elu: {0:f}".format(loss_score3))

# Predictions
y_predict = regressor.predict(input_fn=lambda: input_fn(test, pred = True))
to_submit(y_predict, "Elu")



#********* Deep Neural Network for continuous and Categorical Features *********

#import and Split
train=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\train.csv")
train.drop('Id', axis=1, inplace=True)
train_numerical=train.select_dtypes(exclude=['object'])
train_numerical.fillna(0, inplace=True)
train_categoric=train.select_dtypes(include=['object'])
train_categoric.fillna('NONE', inplace=True)
train=train_numerical.merge(train_categoric, left_index=True, right_index=True)


test=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\House Price\\test.csv")
ID=test.Id
test.drop('Id', axis=1, inplace=True)
test_numerical=test.select_dtypes(exclude=['object'])
test_numerical.fillna(0, inplace=True)
test_categoric=test.select_dtypes(include=['object'])
test_categoric.fillna('NONE', inplace=True)
test=test_numerical.merge(test_categoric, left_index=True, right_index=True)

#Remove the outliers
from sklearn.ensemble import IsolationForest

clf=IsolationForest(max_samples=100, random_state=42)
clf.fit(train_numerical)
y_noano=clf.predict(train_numerical)
y_noano=pd.DataFrame(y_noano, columns=['Top'])
y_noano[y_noano['Top']==1].index.values

train_numerical=train_numerical.iloc[y_noano[y_noano['Top']==1].index.values]
train_numerical.reset_index(drop=True, inplace=True)

train_categoric=train_categoric.iloc[y_noano[y_noano['Top']==1].index.values]
train_categoric.reset_index(drop=True, inplace=True)

train=train.iloc[y_noano[y_noano['Top']==1].index.values]
train.reset_index(drop=True, inplace=True)

col_train_num= list(train_numerical.columns)
col_train_num_bis=list(train_numerical.columns)

col_train_cat=list(train_categoric.columns)

col_train_num_bis.remove('SalePrice')

mat_train=np.matrix(train_numerical)
mat_test=np.matrix(test_numerical)
mat_new=np.matrix(train_numerical.drop('SalePrice', axis=1))
mat_y=np.array(train.SalePrice)


prepro_y=MinMaxScaler()
prepro_y.fit(mat_y.reshape(1314, 1))

prepro=MinMaxScaler()
prepro.fit(mat_train)

prepro_test=MinMaxScaler()
prepro_test.fit(mat_new)

train_num_scale=pd.DataFrame(prepro.transform(mat_train), columns=col_train)
test_num_scale=pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_bis)

train[col_train_num]=pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
test[col_train_num_bis]=test_num_scale

#List of features
COLUMNS=col_train_num
FEATURES=col_train_num_bis
LABEL="SalePrice"

FEATURES_CAT=col_train_cat

engineered_features=[]


for continuous_feature in FEATURES:
    engineered_features.append(
            tf.contrib.layers.real_valued_column(continuous_feature))


for categorical_feature in FEATURES_CAT:
    sparse_column=tf.contrib.layers.sparse_column_with_hash_bucket(
            categorical_feature, hash_bucket_size=1000)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column,
                                                                  dimension=16, combiner="sum"))


#Training Set and Prediction Set with the features to predict

training_set=train[FEATURES+FEATURES_CAT]
prediction_set=train.SalePrice

#Train and Test
x_train, x_test, y_train, y_test= train_test_split(training_set[FEATURES+FEATURES_CAT],
                                                   prediction_set, test_size=0.33, 
                                                   random_state=42)

y_train=pd.DataFrame(y_train, columns=[LABEL])
training_set=pd.DataFrame(x_train, columns=FEATURES+FEATURES_CAT).merge(y_train, left_index=True, right_index=True)

#Training for submission
training_sub=training_set[FEATURES+FEATURES_CAT]
testing_sub=test[FEATURES+FEATURES_CAT]

#Same thing but for the Test set
y_test=pd.DataFrame(y_test, columns=[LABEL])
testing_set=pd.DataFrame(x_test, columns=FEATURES+FEATURES_CAT).merge(y_test, left_index=True,
                        right_index=True)


training_set[FEATURES_CAT]=training_set[FEATURES_CAT].applymap(str)
testing_set[FEATURES_CAT]=testing_set[FEATURES_CAT].applymap(str)


def input_fn_new(data_set, training=True):
    continuous_cols={k:tf.constant(data_set[k].values) for k in FEATURES}
    
    categorical_cols={k:tf.SparseTensor(
            indices =[[i,0] for i in range(data_set[k].size)], values=data_set[k].values,
            dense_shape=[data_set[k].size,1]) for k in FEATURES_CAT}
    
    #Merge the two dictionaries into one
    feature_cols=dict(list(continuous_cols.items())+ list(categorical_cols.items()))
    
    if training==True:
        #Converts the level column into  a constant Tensor
        label=tf.constant(data_set[LABEL].values)
        
        #Returns the feature column and the label
        return feature_cols, label
    
    return feature_cols



#Model
regressor=tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                        activation_fn=tf.nn.relu, hidden_units=
                                        [500,250, 120, 60, 30])


categorical_cols={k: tf.SparseTensor(indices=[[i,0] for i in range(training_set[k].size)],
    values=training_set[k].values, dense_shape=[training_set[k].size,1]) for k in FEATURES_CAT}



#Deep Neural Network Regressor with the Trining set which contain the data split by train and test split
regressor.fit(input_fn=lambda:input_fn_new(training_set), steps=3500)


ev=regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training=True), steps=1 )


loss_score4=ev["loss"]
print("Final loss on the Testing set:  {0:f}".format(loss_score4))


#Predictions bis

#Predictions
y=regressor.predict(input_fn=lambda: input_fn_new(testing_set))
predictions=list(itertools.islice(y, testing_set.shape[0]))
predictions=pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(434,1)))

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

fig, ax=plt.subplots(figsize=(5,5))
plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize=10)
plt.ylabel('Reality', fontsize=10)
plt.title("Prediction X Reality on dataset Test", fontsize=10)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()


y_predict=regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training=False))
to_submit(y_predict, "submission_cont_categ")



#******************* SHALLOW NETWORK ***********************

# Model
regressor = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 
                                          activation_fn = tf.nn.relu, hidden_units=[2500])

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn = lambda: input_fn_new(training_set) , steps=3500)

ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)
loss_score5 = ev["loss"]


print("Final Loss on the testing set: {0:f}".format(loss_score5))



y_predict = regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training = False))    
to_submit(y_predict, "submission_shallow")


#Conclusion
list_score=[loss_score1,loss_score2, loss_score3, loss_score4, loss_score5]
list_model=['Relu_cont', 'LRelu_cont', 'ELU_cont', 'Relu_cont_cat', 'Shallow_lku']

import matplotlib.pyplot as plt; plt.rcdefaults()

plt.style.use('ggplot')
objects=list_model
y_pos=np.arange(len(objects))
performance=list_score

plt.barh(y_pos, performance, align='center', alpha=0.9)
plt.yticks(y_pos, objects)
plt.xlabel('Loss')
plt.title('Model comparision')
plt.show()










