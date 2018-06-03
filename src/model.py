
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
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

# Local
import utils
import deb
np.set_printoptions(suppress=True)


class conv_lstm(object):
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
		self.conf={'mode':1}
		self.debug=1
		#utils.conf["stage"]="train"
		self.model_build()
	def model_build(self):
		self.data = tf.placeholder(tf.float32, [None] +[self.timesteps] + self.shape + [self.channels])
		self.target = tf.placeholder(tf.float32, [None, self.n_classes])
		if self.debug: deb.prints(self.target.get_shape)

		self.prediction = self.model_graph_get(self.data)

		# Set optimizer
		self.minimize=self.loss_optimizer_set(self.target,self.prediction)
		self.mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
		self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
		
		self.error_sum = tf.summary.scalar("error", self.error)

		t_vars = tf.trainable_variables()

		self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
		self.merged = tf.summary.merge_all()

		if self.debug: print("trainable parameters",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

	def train(self, args):
		#init_op = tf.initialize_all_variables()
		init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init_op)
#		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], graph=tf.get_default_graph())
		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], self.sess.graph)

		data = self.data_load(utils.conf)
		batch_idxs = min(len(data["train"]["im_paths"]), args.train_size) // self.batch_size
		deb.prints(data["train"]["labels"].shape)
		deb.prints(data["test"]["labels"].shape)
		deb.prints(batch_idxs)
		 
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
				incorrect = self.sess.run(self.error,{self.data: data["test"]["ims"], self.target: data["test"]["labels"]})
				print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * incorrect))
			if np.mod(epoch, 2) == 0:
				save_path = self.saver.save(self.sess, "./model.ckpt")
				print("Model saved in path: %s" % save_path)
				
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs,time.time() - start_time))

			print("Epoch - {}. Steps per epoch - {}".format(str(epoch),str(idx)))
			
	def test(self, args):
		self.sess = tf.Session()
		self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))
		print("Model restored.")
		data = self.data_load(utils.conf)
		batch_idxs = min(len(data["train"]["im_paths"]), args.train_size) // self.batch_size
		idx=1
		batch_file_paths = data["train"]["im_paths"][idx*self.batch_size:(idx+1)*self.batch_size]
		data["test"]["ims"] = np.asarray(data["test"]["ims"])
		data["train"]["ims"] = np.asarray([np.load(batch_file_path) for batch_file_path in batch_file_paths]) # Load files from path
		deb.prints(data["train"]["ims"].shape)
		data["train"]["labels"] = data["train"]["labels"][idx*self.batch_size:(idx+1)*self.batch_size]
		self.model_test_on_samples(data)

	def model_test_on_samples(self,dataset):
		print("train results")
		count=1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["train"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["train"]["labels"][count])
		count=count+1
			
		print("test results")
		count=1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["test"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["test"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["test"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["test"]["labels"][count])
		count=count+1
		print(np.around(self.sess.run(self.prediction,{self.data: np.expand_dims(dataset["test"]["ims"][count,:,:,:,:],axis=0)}),decimals=5))
		deb.prints(dataset["test"]["labels"][count])
		count=count+1

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

	def layer_lstm_get(self,data,filters,kernel):
		#filters=64
		cell = tf.contrib.rnn.ConvLSTMCell(2,self.shape + [self.channels], filters, kernel)
		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		if self.debug: deb.prints(val.get_shape)
		last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
		if self.debug: deb.prints(last.get_shape())
		return last


	def loss_optimizer_set(self,target,prediction):
		# Estimate loss from prediction and target
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

		# Prepare the optimization function
		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(cross_entropy)
		return minimize



