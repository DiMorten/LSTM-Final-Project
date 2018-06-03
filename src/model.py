

from __future__ import division
import os
import math
import json
import random
import pprint
import time
import scipy.misc
import numpy as np
from time import gmtime, strftime
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

# Local
import utils
import deb
np.set_printoptions(suppress=True)

# ===================================NeuralNet generic class ======================================================= #
# =================================== Might take onehot or image output ============================================= #
class NeuralNet(object):

	def __init__(self, sess, batch_size=50, epoch=200, train_size=1e8,
                        timesteps=utils.conf["t_len"], shape=[32,32],
                        kernel=[3,3], channels=6, filters=32, n_classes=9,
                        checkpoint_dir='./checkpoint'):
		self.sess = sess
		self.batch_size = batch_size
		self.epoch = epoch
		self.train_size = train_size
		self.timesteps = timesteps
		self.shape = shape
		self.kernel = kernel
		self.kernel_size = kernel[0]
		self.channels = channels
		self.filters = filters
		self.n_classes = n_classes
		self.checkpoint_dir = checkpoint_dir
		self.conf=utils.conf
		self.debug=1

	def layer_lstm_get(self,data,filters,kernel):
		#filters=64
		cell = tf.contrib.rnn.ConvLSTMCell(2,self.shape + [self.channels], filters, kernel)
		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		if self.debug: deb.prints(val.get_shape)
		last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
		if self.debug: deb.prints(last.get_shape())
		return last

	def tensorboard_saver_init(self, error):
		error_sum = tf.summary.scalar("error", error)		
		saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
		merged = tf.summary.merge_all()
		return error_sum, saver, merged
	def trainable_vars_print(self):
		t_vars = tf.trainable_variables()
		if self.debug: print("trainable parameters",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
# ============================ NeuralNet takes onehot image output ============================================= #
class NeuralNetOneHot(NeuralNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def average_accuracy_get(self,target,prediction):
		correct_per_class=np.zeros(self.n_classes).astype(np.float32)
		for clss in range(0,self.n_classes):
			correct_per_class[clss]=float(np.count_nonzero(np.equal(target[:,clss],prediction[:,clss])))/float(target.shape[0])
		correct_per_class_average = np.average(correct_per_class)

		return correct_per_class_average
	def loss_optimizer_set(self,target,prediction):
		# Estimate loss from prediction and target
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

		# Prepare the optimization function
		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(cross_entropy)

		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		return minimize, mistakes, error
	def train(self, args):
		#init_op = tf.initialize_all_variables()
		init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init_op)
#		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], graph=tf.get_default_graph())
		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], self.sess.graph)

		data = self.data_load(self.conf)
		batch_idxs = min(len(data["train"]["im_paths"]), args.train_size) // self.batch_size
		deb.prints(data["train"]["labels"].shape)
		deb.prints(data["test"]["labels"].shape)
		deb.prints(batch_idxs)
		data["train"]["labels_int"]=[ np.where(r==1)[0][0] for r in data["train"]["labels"] ]
		print("train classes",np.unique(data["train"]["labels_int"],return_counts=True))
		 
		counter = 1
		start_time = time.time()
		for epoch in range(args.epoch):
			#np.random.shuffle(data)
			for idx in range(0, batch_idxs):
				batch_file_paths = data["train"]["im_paths"][idx*self.batch_size:(idx+1)*self.batch_size]
				batch_labels = data["train"]["labels"][idx*self.batch_size:(idx+1)*self.batch_size]
				batch_images = [np.load(batch_file_path) for batch_file_path in batch_file_paths] # Load files from path

				summary,_ = self.sess.run([self.merged,self.minimize],{self.data: batch_images, self.target: batch_labels})
				self.writer.add_summary(summary, counter)
				counter += 1
				self.incorrect = self.sess.run(self.error,{self.data: data["test"]["ims"], self.target: data["test"]["labels"]})
				print('Epoch {:2d}, step {:2d}. Overall accuracy {:3.1f}%'.format(epoch + 1, idx, 100 - 100 * self.incorrect))
			save_path = self.saver.save(self.sess, "./model.ckpt")
			print("Model saved in path: %s" % save_path)
			
			prediction = np.around(self.sess.run(self.prediction,{self.data: data["test"]["ims"]}),decimals=2)
			average_accuracy = self.average_accuracy_get(data["test"]["labels"],prediction)
			deb.prints(average_accuracy)
	
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs,time.time() - start_time))

			print("Epoch - {}. Steps per epoch - {}".format(str(epoch),str(idx)))
			
	def test(self, args):
		self.sess = tf.Session()
		self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))


		print("Model restored.")
		data = self.data_load(self.conf)
		batch_idxs = min(len(data["train"]["im_paths"]), args.train_size) // self.batch_size
		idx=1
		batch_file_paths = data["train"]["im_paths"][idx*self.batch_size:(idx+1)*self.batch_size]
		data["test"]["ims"] = np.asarray(data["test"]["ims"])
		data["train"]["ims"] = np.asarray([np.load(batch_file_path) for batch_file_path in batch_file_paths]) # Load files from path
		deb.prints(data["train"]["ims"].shape)
		data["train"]["labels"] = data["train"]["labels"][idx*self.batch_size:(idx+1)*self.batch_size]
		
		prediction = np.around(self.sess.run(self.prediction,{self.data: data["test"]["ims"]}),decimals=2)
		deb.prints(data["test"]["labels"])
		average_accuracy = self.average_accuracy_get(data["test"]["labels"],prediction)
		deb.prints(average_accuracy)
		self.model_test_on_samples(data)

	def model_test_on_samples(self,dataset,sample_range=range(15,20)):

		print("train results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["train"]["ims"][sample_range]}),decimals=4))
		deb.prints(dataset["train"]["labels"][sample_range])
		
		print("test results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["test"]["ims"][sample_range]}),decimals=4))
		deb.prints(dataset["test"]["labels"][sample_range])

	def data_load(self, conf):

		data={}
		data["train"]={}
		data["test"]={}
		data["train"]["im_paths"] = glob.glob(conf["train"]["balanced_path_ims"]+'/*.npy')
		data["train"]["im_paths"] = sorted(data["train"]["im_paths"], key=lambda x: int(x.split('_')[1][:-4]))

		#print(data["train"]["im_paths"])
		data["test"]["im_paths"] = glob.glob(conf["test"]["balanced_path_ims"]+'/*.npy')
		data["test"]["im_paths"] = sorted(data["test"]["im_paths"], key=lambda x: int(x.split('_')[1][:-4]))

		deb.prints(len(data["train"]["im_paths"]))
		
		data["train"]["labels"] = np.load(conf["train"]["balanced_path_label"]+"labels.npy")
		
		data["test"]["labels"] = np.load(conf["test"]["balanced_path_label"]+"labels.npy")
		data["test"]["ims"]=[np.load(im_path) for im_path in data["test"]["im_paths"]]
		return data

# ================================= Implements ConvLSTM ============================================== #
class conv_lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()

	def model_build(self):
		self.data,self.target=self.placeholder_init(self.timesteps,self.shape,self.channels,self.n_classes)
		self.prediction = self.model_graph_get(self.data)

		# Set optimizer
		self.minimize,self.mistakes,self.error=self.loss_optimizer_set(self.target,self.prediction)
		self.error_sum, self.saver, self.merged = self.tensorboard_saver_init(self.error)
		self.trainable_vars_print()
		
	def placeholder_init(self,timesteps,shape,channels,n_classes):
		data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels])
		target = tf.placeholder(tf.float32, [None, n_classes])
		if self.debug: deb.prints(target.get_shape())
		return data,target
	def model_graph_get(self,data):
		graph_pipeline=self.layer_lstm_get(data,filters=self.filters,kernel=self.kernel)
		
		if self.debug: deb.prints(graph_pipeline.get_shape())
		#graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh)
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, 128,activation=tf.nn.tanh)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return graph_pipeline


