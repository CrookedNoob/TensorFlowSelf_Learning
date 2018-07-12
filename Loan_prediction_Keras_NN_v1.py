# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:07:39 2018

@author: soumyama
"""

#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import urllib.request as request
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
pd.set_option('display.expand_frame_repr', False)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics




#Import dataset
TRAIN_Path="C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\AV\\LPIII\\train.csv"
TEST_path= "C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\AV\\LPIII\\test.csv"   

train=pd.read_csv(TRAIN_Path)
train.head()
test=pd.read_csv(TEST_path)
test.head()

#Check the dimensions and features
print("\nThe train data before dropping the Loan_ID: {}".format(train.shape) )
print("\nThe test data before dropping the Loan_ID: {}".format(test.shape) )

#Save Loan_ID separately
train_ID=train.Loan_ID
test_ID=test.Loan_ID
train_ID.head()

#Drop Loan ID
train.drop("Loan_ID", axis=1, inplace=True)
test.drop("Loan_ID", axis=1, inplace=True)

#Check the dimensions and features after dropping Loan_ID
print("\nThe train data before dropping the Loan_ID: {}".format(train.shape) )
print("\nThe test data before dropping the Loan_ID: {}".format(test.shape) )

#Data Processing

#Name of columns with missing values
train.columns[train.isnull().any()]
test.columns[test.isnull().any()]



#Outliers
sns.distplot(train['ApplicantIncome'], fit=norm)
sns.distplot(test['ApplicantIncome'], fit=norm)
train['ApplicantIncome'].max()
train['ApplicantIncome'].min()

test['ApplicantIncome'].max()
test['ApplicantIncome'].min()


sns.distplot(train['CoapplicantIncome'], fit=norm)
sns.distplot(test['CoapplicantIncome'], fit=norm)

train.LoanAmount.dtype

#Store Label separately
label=train.Loan_Status
label.head()
#Drop Loan_Status from train
train.drop('Loan_Status', axis=1, inplace=True)

#Check Shape
print("\nThe train data before dropping the Loan_ID: {}".format(train.shape) )
print("\nThe test data before dropping the Loan_ID: {}".format(test.shape) )

#We will combine both the train and test data together for imputation
all_data=pd.concat((train,test)).reset_index(drop=True)

#Store the row numbers for both train and test to separate the data later
ntrain=train.shape[0]
ntest=test.shape[0]
ntest

#Imputing categorical data

for col in ('Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term'):
        all_data[col]=all_data[col].fillna(all_data[col].mode())


#We will use the numpy fuction log1p which  applies log(1+x) to all elements of the LoanAmount
all_data["LoanAmount"] = np.log1p(all_data["LoanAmount"])

all_data["LoanAmount"].min()
all_data["LoanAmount"].max()


all_data["LoanAmount"] = all_data["LoanAmount"].fillna(all_data["LoanAmount"].mean())

#We will use the numpy fuction log1p which  applies log(1+x) to all elements of the "ApplicantIncome"
all_data["ApplicantIncome"] = np.log1p(all_data["ApplicantIncome"])

all_data["ApplicantIncome"].min()
all_data["ApplicantIncome"].max()

sns.distplot(train["ApplicantIncome"] , fit=norm)
sns.distplot(all_data["ApplicantIncome"] , fit=norm)

all_data["ApplicantIncome"] = all_data["ApplicantIncome"].fillna(all_data["ApplicantIncome"].mean())


#We will use the numpy fuction log1p which  applies log(1+x) to all elements of the "ApplicantIncome"
all_data["CoapplicantIncome"] = np.log1p(all_data["CoapplicantIncome"])

all_data["CoapplicantIncome"].min()
all_data["CoapplicantIncome"].max()

sns.distplot(train["CoapplicantIncome"] , fit=norm)
sns.distplot(all_data["CoapplicantIncome"] , fit=norm)

all_data["CoapplicantIncome"] = all_data["CoapplicantIncome"].fillna(all_data["CoapplicantIncome"].mean())



#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(all_data["LoanAmount"], plot=plt)
plt.show()

#One-hot encoding for the categorical variables
all_data = pd.get_dummies(all_data)
print(all_data.shape)

#Print the names of variables created
all_data.dtypes.index


#Check the variable type
all_data['ApplicantIncome'].dtype
all_data['LoanAmount'].dtype
all_data['Property_Area_Urban'].dtype


#Separate the train and test data
train = all_data[:ntrain]
test = all_data[ntrain:]

#One-hot encoding Label- Loan Status
label = pd.get_dummies(label)
label.dtypes.index
label.head()

train.shape


#Split Train  and Labels data in to Train_model and Validation data
train_X=train[:600]
train_X.shape
test_X=train[600:]
test_X.shape

train_Y=label[:600]
train_Y.shape
test_Y=label[600:]
test_Y.shape

#Check if any missing values are untreated. If so impute them
train.columns[train.isnull().any()]
test.columns[test.isnull().any()]

train["Loan_Amount_Term"] = train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median())
test["Loan_Amount_Term"] = test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].median())
train["Credit_History"] = train["Credit_History"].fillna(0)
test["Credit_History"] = test["Credit_History"].fillna(0)




import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
#create model

model= Sequential()
model.add(Dense(100, input_dim=20,  activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# fit the model
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), nb_epoch=1000, batch_size=30)


# evaluate the model
scores = model.evaluate(test_X, test_Y)
print ("Accuracy: %.2f%%" %(scores[1]*100))

prediction=model.predict(test)
prediction
np.savetxt("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\AV\\LPIII\\op.csv", prediction, delimiter=",")
