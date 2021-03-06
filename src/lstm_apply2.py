 
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
#from osgeo import gdal
import glob
from skimage.transform import resize
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt

import tensorflow as tf


# Define task: given a binary string of length 20, 
# determine the count of 1s in a binary string.
# For example 01010010011011100110 has 11 ones

import numpy as np
from random import shuffle

#from tensorflow.contrib.rnn import ConvLSTMCell

import utils
import glob

batch_size = 10
timesteps = 9
shape = [32, 32]
kernel = [3, 3]
channels = 6
filters = 12
n_classes=9


data_dim=(9,32,32,6)
ims={}
print(utils.conf)
conf={'mode':1}

#ims=data_load(conf,ims)
#print(len(ims["full"]))
def data_create():
	#train_input=np.ones((58*65,9,32,32,6))
	#train_input=np.ones((20,9,32,32,6)) #10 multitemporal samples
	#train_output=np.ones((20,8))
	n=100
	train_input=np.random.randint(256,size=(n,9,32,32,6)) #10 multitemporal samples
	train_output=np.random.randint(2,size=(n,8))
	
	return n,train_input, train_output

# Split train test data. We have 1M sequences. train 10000

def data_split(train_input, train_output):
	NUM_EXAMPLES = 50
	test_input = train_input[NUM_EXAMPLES:,:,:,:,:]
	test_output = train_output[NUM_EXAMPLES:,:] #everything beyond 10,000
	 
	train_input = train_input[:NUM_EXAMPLES,:,:,:,:]
	train_output = train_output[:NUM_EXAMPLES,:] #till 10,000
	return NUM_EXAMPLES,train_input,train_output,test_input,test_output

def model_define(debug=1):

	# Input data: 20x1 is the string sequence length
	data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels])
	print("data",data.get_shape())
	cell = tf.contrib.rnn.ConvLSTMCell(2,shape + [channels], filters, kernel)

	val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

	target = tf.placeholder(tf.float32, [None, n_classes])
	print("target",target.get_shape())
	n_hidden = 100


	"""
	print("data",data.get_shape())
	# Desired target: batch size X 21 (21 classes)
	target = tf.placeholder(tf.float32, [None, 8])
	print("target",target.get_shape())

	
	cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
	#if debug:
	#    print("cell",tf.shape(cell))
	# Every training example, we are presenting it with the whole
	# sequence
	val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
	"""
	if debug:
		print("val",val.get_shape())
		#print("state",state.get_shape())

	# Apparently transpose output and then 
	# take the output at sequence's last input
	#val = tf.transpose(val, [1, 0, 2])
	if debug: print("val_transpose",val.get_shape())
	last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
	if debug: print("last",last.get_shape())


	fc1 = tf.contrib.layers.flatten(last)
	#shape1 = last.get_shape().as_list()	
	#fc1 = tf.reshape(last,[-1, shape1[1] , shape1[2] * shape1[3]])
	print(fc1.shape)
	print("fc1",fc1)
	fc1 = tf.layers.dense(fc1, n_hidden,activation=tf.nn.tanh)
	if debug: print("fc1",fc1.get_shape())
	prediction = tf.layers.dense(fc1, n_classes,activation=tf.nn.softmax)
	if debug: print("prediction",prediction.get_shape())
	
	"""
	# Weights dimensions num_hidden X number_of_classes (21), thus 
	# when multiplying with the output (val) the resulting dimension
	# wil be batch_size X number_of_classes which is what we are looking
	# for
	weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
	if debug: print("weigth",weight.get_shape())
	bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
	if debug: print("bias",bias.get_shape())
	prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
	if debug: print("prediction",prediction.get_shape())
	"""
	return data,target,prediction

def loss_optimizer_set(target,prediction):
	# Estimate loss from prediction and target
	cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

	# Prepare the optimization function
	optimizer = tf.train.AdamOptimizer()
	minimize = optimizer.minimize(cross_entropy)
	return minimize

def sess_run_train(n,minimize,error,data,train_input,train_output,test_input,test_output):
	# Done designing. Execute model:
	init_op = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init_op)

	# Begin training process
	batch_size = 10
	no_of_batches = int((n_train)/batch_size)
	epoch = 10
	for i in range(epoch):
		ptr = 0
		for j in range(no_of_batches):
			print("ptr,epoch_i,j",ptr,i,j,no_of_batches)
			inp, out = train_input[ptr:ptr+batch_size,:,:,:,:], train_output[ptr:ptr+batch_size,:]
			ptr+=batch_size
			sess.run(minimize,{data: inp, target: out})
			print("Step - ",str(j))
		print("Epoch - ",str(i))
	incorrect = sess.run(error,{data: test_input, target: test_output})
	print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

	# One single string
	#print sess.run(model.prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
	sess.close()

if __name__ == "__main__":
	if conf["mode"]==1:
		n,X,y=data_create()    
		n_train,train_input,train_output,test_input,test_output=data_split(X, y)
		## Design the model 
		data,target,prediction=model_define()
		minimize=loss_optimizer_set(target,prediction)
		# Calculating the error on test data
		# Count of how many sequences in the test dataset were classified
		# incorrectly. 
		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

		#dataset=utils.im_patches_npy_multitemporal_from_npy_from_folder_load(utils.conf,1)
		sess_run_train(n,minimize,error,data,train_input,train_output,test_input,test_output)
		#sess_run_train(dataset["train"]["n"],minimize,error,data,dataset["train"]["ims"],train_output,test_input,test_output)