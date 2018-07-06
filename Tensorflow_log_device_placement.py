# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:58:54 2018

@author: soumyama
"""

import os
import tensorflow as tf

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


#Test a simple computation
tf.Session()

with tf.device('/cpu:0'):
    a=tf.constant([1,2,3,4,5,6], shape=[2,3])
    b=tf.constant([1,2,3,4,5,6], shape=[3,2])

c=tf.matmul(a,b)

#Create a session with log_device_placement set to True

sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))


