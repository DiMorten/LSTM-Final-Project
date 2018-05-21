 
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from osgeo import gdal
import glob
from skimage.transform import resize
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

import tensorflow as tf


# Define task: given a binary string of length 20, 
# determine the count of 1s in a binary string.
# For example “01010010011011100110” has 11 ones

import numpy as np
from random import shuffle
 
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

# Generate output data for every input

train_output = []
 
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

# Split train test data. We have 1M sequences. train 10000

NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 10,000

## Design the model 

# Input data: 20x1 is the string sequence length
data = tf.placeholder(tf.float32, [None, 20,1])

# Desired target: batch size X 21 (21 classes)
target = tf.placeholder(tf.float32, [None, 21])

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# Apparently transpose output and then 
# take the output at sequence's last input
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# Weights dimensions num_hidden X number_of_classes (21), thus 
# when multiplying with the output (val) the resulting dimension
# wil be batch_size X number_of_classes which is what we are looking
# for
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

# Estimate loss from prediction and target
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

# Prepare the optimization function
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

# Calculating the error on test data
# Count of how many sequences in the test dataset were classified
# incorrectly. 
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Done designing. Execute model:
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

# Begin training process
batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print("Epoch - ",str(i))
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

# One single string
#print sess.run(model.prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
sess.close()

