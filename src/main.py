
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
import numpy as np
from random import shuffle
#from tensorflow.contrib.rnn import ConvLSTMCell
import glob
import sys
import pickle
import argparse

# Local
import utils
import deb
#import conf
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='20160419', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=50, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--timesteps', dest='timesteps', default=utils.conf["t_len"], help='# timesteps used to train')
parser.add_argument('--shape', dest='shape', default=[32,32], help='# timesteps used to train')
parser.add_argument('--kernel', dest='kernel', default=[3,3], help='# timesteps used to train')
parser.add_argument('--channels', dest='channels', default=6, help='# timesteps used to train')
parser.add_argument('--filters', dest='filters', default=32, help='# timesteps used to train')
parser.add_argument('--n_classes', dest='n_classes', default=9, help='# timesteps used to train')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
args = parser.parse_args()
np.set_printoptions(suppress=True)

with open(utils.conf["path"]+'data.pkl', 'rb') as handle: dataset=pickle.load(handle)
deb.prints(dataset["train"]["ims"].shape)
deb.prints(dataset["train"]["labels_onehot"].shape)
deb.prints(dataset["test"]["ims"].shape)
deb.prints(dataset["test"]["labels_onehot"].shape)
args.train_size = dataset["train"]["ims"].shape[0]

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    with tf.Session() as sess:
        model = conv_lstm(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                        timesteps=args.timesteps, shape=args.shape,
                        kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                        checkpoint_dir=args.checkpoint_dir)

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'generate_image':
            model.generate_image(args)
        elif args.phase == 'create_dataset':
            model.create_dataset(args)
        else:
            print ('...')


if __name__ == '__main__':
    tf.app.run()






batch_size = 10
timesteps = utils.conf["t_len"]
shape = [32, 32]
kernel = [3, 3]
channels = 6
filters = 12
n_classes=9

np.set_printoptions(suppress=True)
data_dim=(9,32,32,6)
ims={}
#print(utils.conf)
conf={'mode':1}
conf["stage"]="train"

#conf["stage"]="test"

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

def model_define(debug=1,rnn_flag=True):

	# Input data: 20x1 is the string sequence length
	data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels])
	print("data",data.get_shape())
	target = tf.placeholder(tf.float32, [None, n_classes])
	if debug: print("target",target.get_shape())

	filters = 32
	
	if rnn_flag:
		cell = tf.contrib.rnn.ConvLSTMCell(2,shape + [channels], filters, kernel)

		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

		if debug: print("val",val.get_shape())
		last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
		if debug: print("last",last.get_shape())
	else:
		last2 = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)

		last=tf.layers.conv2d(last2, 32, 3, activation=tf.nn.tanh)

	pool1 = tf.layers.max_pooling2d(inputs=last, pool_size=[2, 2], strides=2)
	##if debug: print("pool1",pool1.get_shape())

	# Convolution Layer with 32 filters and a kernel size of 5
	conv1 = tf.layers.conv2d(pool1, 32, 3, activation=tf.nn.tanh)

	fc1 = tf.contrib.layers.flatten(conv1)

	print(fc1.shape)
	print("fc1",fc1)
	prediction = tf.layers.dense(fc1, n_classes,activation=tf.nn.softmax)
	if debug: print("prediction",prediction.get_shape())

	return data,target,prediction

def loss_optimizer_set(target,prediction):
	# Estimate loss from prediction and target
	cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

	# Prepare the optimization function
	optimizer = tf.train.AdamOptimizer()
	minimize = optimizer.minimize(cross_entropy)
	return minimize

def sess_run_train(n_train,minimize,error,data,train_input,train_output,test_input,test_output,debug=1):
	fname=sys._getframe().f_code.co_name
	if conf["stage"]=="train":
		# Done designing. Execute model:
		init_op = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init_op)
		writer = tf.summary.FileWriter(utils.conf["summaries_path"], graph=tf.get_default_graph())

		# Begin training process
		#batch_size = 752
		#batch_size = 300
		#batch_size = 150
		#batch_size = 50
		if utils.conf["pc_mode"]=="remote":
			#batch_size=314
			batch_size=157
			epoch = 300
			
		else:
			batch_size=210
			epoch = 30
		batch_size=50
		epoch = 200
		print("n_train",n_train)
		no_of_batches = int(np.round(float(n_train)/float(batch_size)))
		#no_of_batches=5
		deb.prints(no_of_batches,fname)
		
		
		deb.prints(epoch)
		deb.prints(train_input.shape)
		deb.prints(train_output.shape)
		
		for i in range(epoch):
			ptr = 0
			for j in range(no_of_batches):
				#print("ptr,epoch_i,j",ptr,i,j,no_of_batches)
				inp, out = train_input[ptr:ptr+batch_size,:,:,:,:], train_output[ptr:ptr+batch_size,:]
				inp_test, out_test = test_input[ptr:ptr+batch_size,:,:,:,:], test_output[ptr:ptr+batch_size,:]
				if debug>=3: print(ptr,inp.shape,out.shape)
				ptr+=batch_size
				if debug>=3: print(ptr,inp.shape,out.shape)
				summary,_ = sess.run([merged,minimize],{data: inp, target: out})
				if debug>=3: print("Step - ",str(j))
			if i%10==0:
				# Save the variables to disk.
				  save_path = saver.save(sess, "./model.ckpt")
				  print("Model saved in path: %s" % save_path)
				  writer.add_summary(summary,i+j)
			print("Epoch - {}. Steps per epoch - {}".format(str(i),str(j)))
			incorrect = sess.run(error,{data: inp_test, target: out_test})
			print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
	elif conf["stage"]=="test":
			sess=tf.Session()
			saver.restore(sess,tf.train.latest_checkpoint('./'))
			#saver.restore(sess, "./model.ckpt")
			print("Model restored.")
			# One single string
	print("train results")
	count=1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["train"]["labels_onehot"][count])
	count=count+1
		
	print("test results")
	count=1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["test"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["test"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["test"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["test"]["labels_onehot"][count])
	count=count+1
	print(np.around(sess.run(prediction,{data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
	deb.prints(dataset["test"]["labels_onehot"][count])
	count=count+1

	sess.close()
data_mode=1
if __name__ == "__main__":
	#utils.conf["subdata"]["n"]=3760
	#utils.conf["subdata"]["n"]=2000
	#utils.conf["subdata"]["n"]=1000
	if conf["mode"]==1:
		## Design the model 
		data,target,prediction=model_define()
		minimize=loss_optimizer_set(target,prediction)
		# Calculating the error on test data
		# Count of how many sequences in the test dataset were classified
		# incorrectly. 
		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		with tf.name_scope('summaries'):
			error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		print("trainable parameters",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
		tf.summary.scalar("errated", error)
		saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
		merged = tf.summary.merge_all()
		
		if data_mode==1:
			#utils.im_patches_npy_multitemporal_from_npy_from_folder_load(utils.conf,1,subdata_flag=utils.conf["subdata"]["flag"],subdata_n=utils.conf["subdata"]["n"])
			#dataset=np.load(utils.conf["path"]+"data.npy")

			with open(utils.conf["path"]+'data.pkl', 'rb') as handle: dataset=pickle.load(handle)
			deb.prints(dataset["train"]["ims"].shape)
			deb.prints(dataset["train"]["labels_onehot"].shape)
			deb.prints(dataset["test"]["ims"].shape)
			deb.prints(dataset["test"]["labels_onehot"].shape)

			test_set_reduce=True
			if test_set_reduce:
				index = range(dataset["test"]["ims"].shape[0])
				test_n=500
				index = np.random.choice(index, test_n, replace=False)
				dataset_test_ims=dataset["test"]["ims"][index]
				dataset_test_labels_onehot=dataset["test"]["labels_onehot"][index]

				dataset["test"]["ims"]=dataset_test_ims
				dataset["test"]["labels_onehot"]=dataset_test_labels_onehot
				del dataset_test_labels_onehot
				del dataset_test_ims
			#sess_run_train(n,minimize,error,data,train_input,train_output,test_input,test_output)
			sess_run_train(dataset["train"]["ims"].shape[0],minimize,error,data,dataset["train"]["ims"],dataset["train"]["labels_onehot"],dataset["test"]["ims"],dataset["test"]["labels_onehot"])
		elif data_mode==0:
			n,X,y=data_create()    
			n_train,train_input,train_output,test_input,test_output=data_split(X, y)
		
			sess_run_train(n_train,minimize,error,data,train_input,train_output,test_input,test_output)