# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:13:39 2018

@author: soumyama
"""

from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("Tensorflow Version: {}".format(tf.VERSION))
print("Eager Execution: {}".format(tf.executing_eagerly()))


train_dataset_url="http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp=tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)


print("Local copy of dataset file : {}".format(train_dataset_fp))

import csv

with open('C:\\Users\\soumyama\\.keras\\datasets\\iris_training.csv', 'rb') as csvfile:
    data=csv.reader(csvfile, delimiter=" ", quotechar="|")
    for rows in data:
        print(", ".join(rows))
        
        
!head -n5 {train_dataset_fp}

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]] #Set field type
  parsed_line = tf.decode_csv(line, example_defaults)
  #Fisrt 4 fields are features, combined into a single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  #Last field is the label
  label=tf.reshape(parsed_line[-1], shape=())
  return features, label


#Create the training tf.data.Dataset
train_dataset=tf.data.TextLineDataset(train_dataset_fp)
train_dataset=train_dataset.skip(1) #Skip the first header row
train_dataset=train_dataset.map(parse_csv) #parse each row
train_dataset=train_dataset.shuffle(buffer_size=1000) #Randomize
train_dataset=train_dataset.batch(32)

#View a single example entry from a batch
features, label =iter(train_dataset).next()
print("Example Features: ", features[2])
print("Example Label: ", label[2])

#Using Keras
model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)), #Input shape is required
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)])

#Define loss and gradient function
def loss(model, x,y):
    y_=model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value=loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

#Create optimizer
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)

#Note: Rerunning the below part uses the same model variables

#Keep resluts for plotting
train_loss_results=[]
train_accuracy_results=[]

num_epochs=201

for epoch in range(num_epochs):
    epoch_loss_avg=tfe.metrics.Mean()
    epoch_accuracy=tfe.metrics.Accuracy()
    
    #Training loop uses batch of 32
    for x, y in train_dataset:
        #Optimize the model
        grads=grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())
        #Track Progress
        epoch_loss_avg(loss(model, x, y))#add current batch loss
        #compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    
    #End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    if epoch % 50 ==0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),
              epoch_accuracy.result()))


#Visualise the loss function over time
fig, axes= plt.subplots(2, sharex=True, figsize=(12,8))
fig.suptitle("Training Metrics")

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)


#Evaluate model's effectiveness

#Set up the test dataset
test_url= "http://download.tensorflow.org/data/iris_test.csv"
test_fp= tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)

test_dataset=tf.data.TextLineDataset(test_fp)
test_dataset=test_dataset.skip(1) #Leave the header row
test_dataset=test_dataset.map(parse_csv) #parse each row with the func created earlier
test_dataset=test_dataset.shuffle(1000) #Shuffle the data to randomize
test_dataset=test_dataset.batch(32) #Use the same batch size as that of training set   

#Evaluate the model on test dataset
test_accuracy= tfe.metrics.Accuracy()

for (x,y) in test_dataset:
    prediction=tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
    
print("Test Accuracy : {:.3%}".format(test_accuracy.result()))


#Use the trained model to amke predictions

# 0: Iris setosa 1: Iris versicolor 2: Iris virginica

class_ids = ["Iris Setosa", "Iris Versicolor", "Iris Virginica" ]
predict_dataset=tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5],
        [5.9, 3.0, 4.2, 1.5],
        [6.9, 3.1, 5.4, 2.1]])

predictions=model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx=tf.argmax(logits).numpy()
    name=class_ids[class_idx]
    print("Example {} Prediction : {}".format(i,name))

     

    
    