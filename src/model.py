

from __future__ import division
import os
import math
#import json
import random
#import pprint
import time
#import scipy.misc
import numpy as np
from time import gmtime, strftime
import glob
#from skimage.transform import resize
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
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

	def __init__(self, sess=tf.Session(), batch_size=50, epoch=200, train_size=1e8,
						timesteps=utils.conf["t_len"], shape=[32,32],
						kernel=[3,3], channels=6, filters=32, n_classes=9,
						checkpoint_dir='./checkpoint',log_dir=utils.conf["summaries_path"],data=None, conf=utils.conf, debug=1):
		
		self.ram_data=data
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
		self.conf=conf
		self.debug=debug
		self.log_dir=log_dir
		self.test_batch_size=100
		if self.debug>=1: print("Initializing NeuralNet instance")
		print(self.log_dir)

	# =_______________ Generic Layer Getters ___________________= #
	def layer_lstm_get(self,data,filters,kernel,name="convlstm",get_last=True):
		#filters=64
		cell = tf.contrib.rnn.ConvLSTMCell(2,self.shape + [self.channels], filters, kernel,name=name)
		
		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		if self.debug: deb.prints(val.get_shape())
		kernel,bias=cell.variables
		#self.hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
		tf.summary.histogram('convlstm', kernel)
		if get_last==True:
			if self.debug: deb.prints(val.get_shape())
			last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
			if self.debug: deb.prints(last.get_shape())
			return last
		else:
			return val

	# =____________________ Debug helpers ___________________= #
	def tensorboard_saver_init(self, error):
		error_sum = tf.summary.scalar("error", error)		
		saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
		merged = tf.summary.merge_all()
		return error_sum, saver, merged
	def trainable_vars_print(self):
		t_vars = tf.trainable_variables()
		if self.debug: print("trainable parameters",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
	# =____________________ Train methods ___________________= #
	def data_len_get(self,data,memory_mode):
		if memory_mode=="hdd":
			data_len=len(data["im_paths"])
		elif memory_mode=="ram":
			data_len=data["ims"].shape[0]
		deb.prints(data_len)
		return data_len
	def ram_batch_ims_labels_get(self,batch,data,batch_size,idx):
		
		batch["ims"] = data["ims"][idx*batch_size:(idx+1)*batch_size]
		batch["labels"] = data["labels"][idx*batch_size:(idx+1)*batch_size]
		return batch
	def data_n_get(self,data,memory_mode):
		if memory_mode=="hdd":
			return len(data["im_paths"])
		elif memory_mode=="ram":
			return data["ims"].shape[0]

	def ram_data_sub_data_get(self, data,n,sub_data):

		sub_data["labels"] = data["labels"][sub_data["index"]]
		sub_data["ims"]=data["ims"][sub_data["index"]]
		return sub_data

	def batch_ims_labels_get(self,batch,data,batch_size,idx,memory_mode):
		if memory_mode=="hdd":
			return self.hdd_batch_ims_labels_get(batch,data,batch_size,idx)
		elif memory_mode=="ram":
			return self.ram_batch_ims_labels_get(batch,data,batch_size,idx)
	def data_sub_data_get(self, data,n,memory_mode):
		sub_data={"n":n}		
		sub_data["index"] = np.random.choice(data["index"], sub_data["n"], replace=False)
		deb.prints(sub_data["index"].shape)

		if memory_mode=="hdd":
			sub_data=self.hdd_data_sub_data_get(data,n,sub_data)
		elif memory_mode=="ram":
			sub_data=self.ram_data_sub_data_get(data,n,sub_data)
		deb.prints(sub_data["ims"].shape)
		return sub_data
	def train_init(self):
		init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init_op)
#		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], graph=tf.get_default_graph())
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
	def train(self, args):
		self.train_init()
		
		data = self.data_load(self.conf,memory_mode=self.conf["memory_mode"])
		deb.prints(data["train"]["n"])
		deb.prints(args.train_size)
		deb.prints(self.batch_size)
		batch={}
		batch["idxs"] = min(data["train"]["n"], args.train_size) // self.batch_size
		if self.debug>=1:
			deb.prints(data["train"]["labels"].shape)
			deb.prints(data["test"]["labels"].shape)
			deb.prints(batch["idxs"])
		
		self.unique_classes_print(data["train"],memory_mode=self.conf["memory_mode"])

		counter = 1
		start_time = time.time()
		if self.data_len_get(data["test"],memory_mode=self.conf["memory_mode"])>1000:
			data["sub_test"]=self.data_sub_data_get(data["test"],1000,memory_mode=self.conf["memory_mode"])
		else:
			data["sub_test"]=data["test"]
		#deb.prints(data["train"]["ims"].shape)
		deb.prints(data["train"]["labels"].shape)
		#deb.prints(data["test"]["ims"].shape)
		deb.prints(data["test"]["labels"].shape)
		

		# =__________________________________ Train in batch. Load images from npy files  _______________________________ = #
		for epoch in range(args.epoch):
			for idx in range(0, batch["idxs"]):
				batch=self.batch_ims_labels_get(batch,data["train"],self.batch_size,idx,memory_mode=self.conf["memory_mode"])
				if self.debug>=3:
					deb.prints(batch["ims"].shape)
					deb.prints(batch["labels"].shape)
				summary,_ = self.sess.run([self.merged,self.minimize],{self.data: batch["ims"], self.target: batch["labels"]})
				self.writer.add_summary(summary, counter)
				counter += 1
				self.incorrect = self.sess.run(self.error,{self.data: data["sub_test"]["ims"], self.target: data["sub_test"]["labels"]})
				if self.debug>=3:
					print('Epoch {:2d}, step {:2d}. Overall accuracy {:3.1f}%'.format(epoch + 1, idx, 100 - 100 * self.incorrect))
			
			# =__________________________________ Test stats get and model save  _______________________________ = #
			save_path = self.saver.save(self.sess, "./model.ckpt")
			print("Model saved in path: %s" % save_path)
			
			stats = self.data_stats_get(data["test"],self.test_batch_size) # For each epoch, get metrics on the entire test set
			
			
			print("Average accuracy:{}, Overall accuracy:{}".format(stats["average_accuracy"],stats["overall_accuracy"]))
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch["idxs"],time.time() - start_time))

			print("Epoch - {}. Steps per epoch - {}".format(str(epoch),str(idx)))
	def data_load(self,conf,memory_mode):
		if memory_mode=="hdd":
			data=self.hdd_data_load(conf)
		elif memory_mode=="ram":
			data=self.ram_data
			data["train"]["n"]=data["train"]["ims"].shape[0]
			data["test"]["n"]=data["test"]["ims"].shape[0]

			deb.prints(self.ram_data["train"]["ims"].shape)
			deb.prints(data["train"]["ims"].shape)

		
		
		data["train"]["index"] = range(data["test"]["n"])
		data["test"]["index"] = range(data["test"]["n"])

		return data
	def model_build(self):
		self.data,self.target=self.placeholder_init(self.timesteps,self.shape,self.channels,self.n_classes)
		self.annotation, self.prediction = self.model_graph_get(self.data)

		self.minimize,self.mistakes,self.error=self.loss_optimizer_set(self.target,self.prediction, self.annotation)
		self.error_sum, self.saver, self.merged = self.tensorboard_saver_init(self.error)
		self.trainable_vars_print()
# ============================ NeuralNetSemantic takes image output ============================================= #

class NeuralNetSemantic(NeuralNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing NeuralNetSemantic instance")
	def placeholder_init(self,timesteps,shape,channels,n_classes):
		data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels])
		target = tf.placeholder(tf.float32, [None] + shape[0::])
		if self.debug: deb.prints(target.get_shape())
		return data,target
	def average_accuracy_get(self,target,prediction,debug=0):
		accuracy_average=0.5
		return accuracy_average
	def loss_optimizer_set(self,target,prediction, annotation):

		target_int=tf.cast(target,tf.int32)
		deb.prints(target_int.get_shape())
		deb.prints(prediction.get_shape())
		#prediction=tf.squeeze(prediction, squeeze_dims=[3])
		#logits=tf.argmax(prediction,1)
		#deb.prints(logits.get_shape())
		
		# Estimate loss from prediction and target
		with tf.name_scope('cross_entropy'):
			cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_int, logits=prediction))

		##with tf.name_scope('learning_rate'):
		##	learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step, opt.iter_epoch, opt.lr_decay, staircase=True)
		# Prepare the optimization function
		##optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
		optimizer = tf.train.AdamOptimizer()

		minimize = optimizer.minimize(cross_entropy)

		annotation=tf.cast(annotation,tf.float32)
		# Distance L1
		error = tf.reduce_sum(tf.cast(tf.abs(tf.subtract(tf.contrib.layers.flatten(annotation),tf.contrib.layers.flatten(target))), tf.float32))
		#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		
		#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		tf.summary.scalar('error',error)
		mistakes=None
		return minimize, mistakes, error

	def unique_classes_print(self,data,memory_mode):
		pass
	def data_stats_get(self,data,batch_size=1000):
		stats={}
		stats["average_accuracy"]=0
		stats["overall_accuracy"]=0
		return stats
# ============================ NeuralNet takes onehot image output ============================================= #
class NeuralNetOneHot(NeuralNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing NeuralNetOneHot instance")

	def placeholder_init(self,timesteps,shape,channels,n_classes):
		data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels])
		target = tf.placeholder(tf.float32, [None, n_classes])
		if self.debug: deb.prints(target.get_shape())
		return data,target
#	def average_accuracy_get(self,target,prediction,correct_per_class_accumulated=False,correct_per_class=None):
	def average_accuracy_get(self,target,prediction,debug=0):	
		correct_per_class_percent = np.zeros(self.n_classes).astype(np.float32)
		correct_per_class = np.zeros(self.n_classes).astype(np.float32)
		targets_int = np.argmax(target,axis=1)
		predictions_int = np.argmax(prediction,axis=1)

		targets_label_count = np.sum(target,axis=0)
		correct_all_classes = targets_int[targets_int == predictions_int]
		count_total = correct_all_classes.shape[0]
		
		if debug>=2: deb.prints(count_total)
		for clss in range(0,self.n_classes):
			correct_per_class[clss]=correct_all_classes[correct_all_classes==clss].shape[0]
		if debug>=2: deb.prints(correct_per_class)
		correct_per_class_average, accuracy_average = self.correct_per_class_average_get(correct_per_class, targets_label_count)
		return correct_per_class_average,correct_per_class,accuracy_average

	def correct_per_class_average_get(self,correct_per_class,targets_label_count):
		correct_per_class_average=np.divide(correct_per_class, targets_label_count)
		accuracy_average=correct_per_class_average[~np.isnan(correct_per_class_average)]
		accuracy_average=accuracy_average[np.nonzero(accuracy_average)]
		accuracy_average=np.average(accuracy_average)
		
		return correct_per_class_average, accuracy_average

	def loss_optimizer_set(self,target,prediction):
		# Estimate loss from prediction and target
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

		# Prepare the optimization function
		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(cross_entropy)

		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
		tf.summary.scalar('error',error)
		return minimize, mistakes, error

	def hdd_batch_ims_labels_get(self,batch,data,batch_size,idx):
		batch["file_paths"] = data["im_paths"][idx*batch_size:(idx+1)*batch_size]
		batch["labels"] = data["labels"][idx*batch_size:(idx+1)*batch_size]
		batch["ims"] = np.asarray([np.load(batch_file_path) for batch_file_path in batch["file_paths"]]) # Load files from path
		return batch



	def ims_get(self,data_im_paths):
		return np.asarray([np.load(file_path) for file_path in data_im_paths]) # Load files from path
		
	def hdd_data_sub_data_get(self, data,n,sub_data):
		
		sub_data["im_paths"] = [data["im_paths"][i] for i in sub_data["index"]]
		sub_data["labels"] = data["labels"][sub_data["index"]]
		sub_data["ims"]=self.ims_get(sub_data["im_paths"])
		return sub_data


	def unique_classes_print(self,data,memory_mode):
		if memory_mode=="hdd":
			data["labels_int"]=[ np.where(r==1)[0][0] for r in data["labels"] ]
			print("Unique classes",np.unique(data["labels_int"],return_counts=True))
		elif memory_mode=="ram":
			print("Unique classes",np.unique(data["labels_int"],return_counts=True))

	def data_stats_get(self,data,batch_size=1000):

		batch={}
		batch["idxs"] = data["n"] // batch_size
		if self.debug>=2:
			deb.prints(batch["idxs"])
			#deb.prints()
		stats={"correct_per_class":np.zeros(self.n_classes).astype(np.float32)}
		stats["per_class_label_count"]=np.zeros(self.n_classes).astype(np.float32)
		
		for idx in range(0, batch["idxs"]):
			batch=self.batch_ims_labels_get(batch,data,batch_size,idx,memory_mode=self.conf["memory_mode"])
			
			batch["prediction"] = np.around(self.sess.run(self.prediction,{self.data: batch["ims"]}),decimals=2)
			batch["label_count"]=np.sum(batch["labels"],axis=0)
			
			if self.debug>=2:
				deb.prints(batch["prediction"].shape)
				deb.prints(batch["labels"].shape)
				deb.prints(batch["label_count"])
			_,batch["correct_per_class"],_=self.average_accuracy_get(batch["labels"],batch["prediction"])
			stats["correct_per_class"]+=batch["correct_per_class"]
			
			if self.debug>=2:
				deb.prints(batch["correct_per_class"])
				deb.prints(stats["correct_per_class"])
			
		if self.debug>=2:
			deb.prints(data["labels"].shape)
		
		stats["per_class_label_count"]=np.sum(data["labels"],axis=0)

		if self.debug>=2:
			deb.prints(stats["correct_per_class"])
			deb.prints(stats["per_class_label_count"])
		

		stats["overall_accuracy"]=np.sum(stats["correct_per_class"][1::])/np.sum(stats["per_class_label_count"][1::])# Don't take backnd (label 0) into account for overall accuracy
		
		stats["per_class_accuracy"],stats["average_accuracy"]=self.correct_per_class_average_get(stats["correct_per_class"][1::], stats["per_class_label_count"][1::])
		if self.debug>=1: 
			deb.prints(stats["overall_accuracy"])
			deb.prints(stats["average_accuracy"])
		if self.debug>=2:
			deb.prints(stats["per_class_accuracy"])
		return stats

	def test(self, args):
		
		self.sess = tf.Session()
		self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))

		print("Model restored.")
		data = self.data_load(self.conf,memory_mode=self.conf["memory_mode"])

		test_stats=self.data_stats_get(data["test"])

	def model_test_on_samples(self,dataset,sample_range=range(15,20)):

		print("train results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["train"]["ims"][sample_range]}),decimals=4))
		deb.prints(dataset["train"]["labels"][sample_range])
		
		print("test results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["test"]["ims"][sample_range]}),decimals=4))
		deb.prints(dataset["test"]["labels"][sample_range])

	def data_group_load(self,conf,data):

		data["im_paths"] = glob.glob(conf["balanced_path_ims"]+'/*.npy')
		data["im_paths"] = sorted(data["im_paths"], key=lambda x: int(x.split('_')[1][:-4]))
		
		data["labels"] = np.load(conf["balanced_path_label"]+"labels.npy")

		data["n"]=len(data["im_paths"])
		data["index"] = range(data["n"])

		return data
		
		
	def hdd_data_load(self, conf):

		data={}
		data["train"]={}
		data["test"]={}
		data["train"]["im_paths"] = glob.glob(conf["train"]["balanced_path_ims"]+'/*.npy')
		data["train"]["im_paths"] = sorted(data["train"]["im_paths"], key=lambda x: int(x.split('_')[1][:-4]))
		data["train"]["n"]=len(data["train"]["im_paths"])
		#print(data["train"]["im_paths"])
		data["test"]["im_paths"] = glob.glob(conf["test"]["balanced_path_ims"]+'/*.npy')
		data["test"]["im_paths"] = sorted(data["test"]["im_paths"], key=lambda x: int(x.split('_')[1][:-4]))

		deb.prints(len(data["train"]["im_paths"]))
		
		data["train"]["labels"] = np.load(conf["train"]["balanced_path_label"]+"labels.npy")
		
		data["test"]["labels"] = np.load(conf["test"]["balanced_path_label"]+"labels.npy")
		
		# Change to a subset of test
		data["test"]["ims"]=[np.load(im_path) for im_path in data["test"]["im_paths"]]
		data["test"]["n"]=len(data["test"]["im_paths"])
		return data


# ================================= Implements U-Net ============================================== #
class UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=10
	def model_graph_get(self,data):
		graph_pipeline = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh,padding='same')
		graph_pipeline = tf.layers.conv2d(graph_pipeline, 9, self.kernel_size, activation=None,padding='same')
		annotation_pred = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), graph_pipeline
		return annotation_pred, graph_pipeline

# ================================= Implements ConvLSTM ============================================== #
class conv_lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		graph_pipeline=self.layer_lstm_get(data,filters=self.filters,kernel=self.kernel,name='convlstm')
		
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh)
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return graph_pipeline


# ================================= Implements Conv3DMultitemp ============================================== #
class Conv3DMultitemp(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		print(1)
		super().__init__(*args, **kwargs)
		self.kernel=[3,3,3]
		deb.prints(self.kernel)
		self.model_build()
		
	def model_graph_get(self,data):
		graph_pipeline=self.layer_lstm_get(data,filters=self.filters,kernel=[3,3],get_last=False,name="convlstm")
		#graph_pipeline=tf.layers.conv3d(graph_pipeline,self.filters,[1,3,3],padding='same',activation=tf.nn.tanh)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline=tf.layers.conv3d(graph_pipeline,16,[3,3,3],padding='same',activation=tf.nn.tanh)
		
		graph_pipeline=tf.layers.max_pooling3d(inputs=graph_pipeline, pool_size=[2,1,1], strides=[2,1,1],padding='same')
		if self.debug: deb.prints(graph_pipeline.get_shape())

		#graph_pipeline=tf.layers.conv3d(graph_pipeline,self.filters,[],padding='same',activation=tf.nn.tanh)
		
		#graph_pipeline=tf.layers.conv3d(data,self.filters,self.kernel,padding='same',activation=tf.nn.tanh)
		
		#graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh)
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, 128,activation=tf.nn.tanh)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return graph_pipeline
