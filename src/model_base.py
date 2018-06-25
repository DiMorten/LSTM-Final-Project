 

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
import cv2
#from cell import ConvGRUCell
#from tf.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

np.set_printoptions(suppress=True)

# ===================================NeuralNet generic class ======================================================= #
# =================================== Might take onehot or image output ============================================= #
class NeuralNet(object):

	def __init__(self, sess=tf.Session(), batch_size=50, epoch=200, train_size=1e8,
						timesteps=utils.conf["t_len"], patch_len=32,
						kernel=[3,3], channels=7, filters=32, n_classes=6,
						checkpoint_dir='./checkpoint',log_dir=utils.conf["summaries_path"],data=None, conf=utils.conf, debug=1, \
						patience=10,squeeze_classes=True,n_repetitions=30):
		self.squeeze_classes=squeeze_classes		
		self.ram_data=data
		self.sess = sess
		self.batch_size = batch_size
		self.epoch = epoch
		self.train_size = train_size
		self.timesteps = timesteps
		self.patch_len = patch_len
		self.shape = [self.patch_len,self.patch_len]
		self.kernel = kernel
		self.kernel_size = kernel[0]
		self.channels = channels
		deb.prints(self.channels)
		self.filters = filters
		self.n_classes = n_classes
		self.checkpoint_dir = checkpoint_dir
		self.conf=conf
		self.debug=debug
		self.log_dir=log_dir
		self.test_batch_size=1000
		self.early_stop={}
		self.early_stop["patience"]=patience
		self.repeat={"n":n_repetitions, "filename": 'repeat_results.pickle'}
		if self.debug>=1: print("Initializing NeuralNet instance")
		print(self.log_dir)

	# =_______________ Generic Layer Getters ___________________= #
	def layer_lstm_get(self,data,filters,kernel,name="convlstm",get_last=True):
		#filters=64
		cell = tf.contrib.rnn.ConvLSTMCell(2,self.shape + [self.channels], filters, kernel,name=name)
		
		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		if self.debug: deb.prints(val.get_shape())
		kernel,bias=cell.variables
		deb.prints(kernel.get_shape())
		#self.hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
		tf.summary.histogram('convlstm', kernel)
		if get_last==True:
			if self.debug: deb.prints(val.get_shape())
			last = tf.gather(val, int(val.get_shape()[1]) - 1,axis=1)
			if self.debug: deb.prints(last.get_shape())
			return last
		else:
			return val
	# =_______________ Generic Layer Getters ___________________= #
	def layer_flat_lstm_get(self,data,filters,kernel,name="convlstm",get_last=True):
		#filters=64
		cell = tf.nn.rnn_cell.LSTMCell(filters,state_is_tuple=True)

		val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
		if self.debug: deb.prints(val.get_shape())
		kernel,bias=cell.variables
		#self.hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
		tf.summary.histogram('convlstm', kernel)
		if get_last==True:
			if self.debug: deb.prints(val.get_shape())
			#val = tf.transpose(val, [1, 0, 2])
			#val = tf.transpose(va)
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
	def train_init(self,args):
		init_op = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init_op)
#		self.writer = tf.summary.FileWriter(utils.conf["summaries_path"], graph=tf.get_default_graph())
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)


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

		
		if self.data_len_get(data["test"],memory_mode=self.conf["memory_mode"])>1000:
			data["sub_test"]=self.data_sub_data_get(data["test"],1000,memory_mode=self.conf["memory_mode"])
		else:
			data["sub_test"]=data["test"]
		#deb.prints(data["train"]["ims"].shape)
		deb.prints(data["train"]["labels"].shape)
		#deb.prints(data["test"]["ims"].shape)
		deb.prints(data["test"]["labels"].shape)
		return batch,data
	def random_shuffle(self,data):
		idxs=np.arange(0,data["n"])
		idxs=np.random.permutation(idxs)
		data["ims"]=data["ims"][idxs]
		data["labels"]=data["labels"][idxs]
		return data

	def data_shuffle(self,data):
		idxs=np.arange(0,data.shape[0])
		idxs=np.random.shuffle(idxs)
		return np.squeeze(data)

	def train_batch_loop(self,args,batch,data):
		start_time = time.time()
		early_stop={"best":{}, "patience":self.early_stop["patience"]}
		early_stop["count"]=0
		early_stop["best"]["metric1"]=0
		counter=1
		# =__________________________________ Train in batch. Load images from npy files  _______________________________ = #
		for epoch in range(args.epoch):
			data["train"]["ims"]=self.data_shuffle(data["train"]["ims"])
			data["train"]["labels"]=self.data_shuffle(data["train"]["labels"])
			
			for idx in range(0, batch["idxs"]):
				batch=self.batch_ims_labels_get(batch,data["train"],self.batch_size,idx,memory_mode=self.conf["memory_mode"])
				if self.debug>=3:
					deb.prints(batch["ims"].shape)
					deb.prints(batch["labels"].shape)
				summary,_ = self.sess.run([self.merged,self.minimize],{self.data: batch["ims"], self.target: batch["labels"], self.keep_prob: 1.0, self.global_step: idx, self.training: True})
				self.writer.add_summary(summary, counter)
				counter += 1
				self.incorrect = self.sess.run(self.error,{self.data: data["sub_test"]["ims"], self.target: data["sub_test"]["labels"], self.keep_prob: 1.0, self.training: True})
				if self.debug>=1 and (idx % 30 == 0):
					print('Epoch {:2d}, step {:2d}. Overall accuracy {:3.1f}%'.format(epoch + 1, idx, 100 - 100 * self.incorrect))
				
			# =__________________________________ Test stats get and model save  _______________________________ = #
			save_path = self.saver.save(self.sess, "./model.ckpt")
			print("Model saved in path: %s" % save_path)
			stats = self.data_stats_get(data["test"],self.test_batch_size) # For each epoch, get metrics on the entire test set
			early_stop=self.early_stop_check(early_stop,stats["overall_accuracy"],stats["average_accuracy"],stats["per_class_accuracy"])
			if early_stop["signal"]:
				deb.prints(early_stop["best"]["metric1"])
				deb.prints(early_stop["best"]["metric2"])
				deb.prints(early_stop["best"]["metric3"])
				
				break
			
			print("Average accuracy:{}, Overall accuracy:{}".format(stats["average_accuracy"],stats["overall_accuracy"]))
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch["idxs"],time.time() - start_time))

			print("Epoch - {}. Steps per epoch - {}".format(str(epoch),str(idx)))
		return early_stop
	def train(self, args):
		batch,data=self.train_init(args)
		
		
		early_stop=self.train_batch_loop(args,batch,data)
	def train_repeat(self,args):
		
		self.repeat["results"]=[]
		with open(self.repeat["filename"], 'wb') as handle:
			pickle.dump(self.repeat, handle, protocol=pickle.HIGHEST_PROTOCOL)
		self.repeat["best_metric1"]=0
		for i in range(self.repeat["n"]):
			batch,data=self.train_init(args)
			early_stop=self.train_batch_loop(args,batch,data)
			self.repeat["results"].append(early_stop)
			print("Repeat step: {}".format(i))
			print("Resultls:",self.repeat["results"])
			
			if early_stop["best"]["metric1"]>self.repeat["best_metric1"]:
				self.repeat["best_metric1"]=early_stop["best"]["metric1"]
				save_path = self.saver.save(self.sess, "./model_best.ckpt")
				print("Best model saved in path: %s" % save_path)

			with open(self.repeat["filename"], 'wb') as handle:
				pickle.dump(self.repeat, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.repeat["best"]={}
		self.repeat["overall_accuracy"]=np.asarray([exp["best"]["metric1"] for exp in self.repeat["results"]])
		self.repeat["average_accuracy"]=np.asarray([exp["best"]["metric2"] for exp in self.repeat["results"]])
	#	self.repeat["npy"]["overall_accuracy"]=np.asarray([exp["best"]["metric1"] for exp in self.repeat["results"]])
		
		deb.prints(self.repeat["overall_accuracy"])
		deb.prints(self.repeat["average_accuracy"])
		self.repeat["best"]["idx"]=np.argmax(self.repeat["overall_accuracy"]["value"])
		self.repeat["best"]["overall_accuracy"]=self.repeat["overall_accuracy"][self.repeat["best"]["idx"]]
		self.repeat["best"]["average_accuracy"]=self.repeat["average_accuracy"][self.repeat["best"]["idx"]]
		
		
		#self.repeat["overall_accuracy"]["mean"]=float(sum(exp["best"]["metric1"] for exp in self.repeat["results"])) / len(self.repeat["results"])
		#self.repeat["average_accuracy"]=float(sum(exp["best"]["metric2"] for exp in self.repeat["results"])) / len(self.repeat["results"])
		self.repeat["per_class_accuracy"]=np.asarray([exp["best"]["metric3"] for exp in self.repeat["results"]])

		print(self.repeat["overall_accuracy"])
		print(self.repeat["average_accuracy"])
		print(self.repeat["per_class_accuracy"])
		
		print(self.repeat)
		with open(self.repeat["filename"], 'wb') as handle:
			pickle.dump(self.repeat, handle, protocol=pickle.HIGHEST_PROTOCOL)
	def repeat_metric_average(self,repeat_results):
		metric_average=float(sum(exp["metric1"] for exp in self.repeat["results"])) / len(self.repeat["results"])

		return float(sum(metric))

	def early_stop_check(self,early_stop,metric1,metric2,metric3):
		early_stop["signal"]=False
		if metric1>early_stop["best"]["metric1"]:
			early_stop["best"]["metric1"]=metric1
			early_stop["best"]["metric2"]=metric2
			early_stop["best"]["metric3"]=metric3
			early_stop["count"]=0
		else:
			early_stop["count"]+=1
			if early_stop["count"]>=early_stop["patience"]:
				early_stop["signal"]=True
			else:
				early_stop["signal"]=False
		return early_stop
	def data_stats_get(self,data,batch_size=1000,im_reconstruct=False):
		batch={}
		batch["idxs"] = data["n"] // batch_size
		if self.debug>=2:
			deb.prints(batch["idxs"])

		stats={"correct_per_class":np.zeros(self.n_classes).astype(np.float32)}
		stats["per_class_label_count"]=np.zeros(self.n_classes).astype(np.float32)
		mask=cv2.imread(self.conf["train"]["mask"]["dir"],0)
		label=cv2.imread("../data/labels/9.tif",0)
		if im_reconstruct:
			self.reconstruct=self.reconstruct_init()

		for idx in range(0, batch["idxs"]):
			batch=self.batch_ims_labels_get(batch,data,batch_size,idx,memory_mode=self.conf["memory_mode"])
			batch["prediction"] = self.batch_prediction_from_sess_get(batch["ims"])
			if self.debug>=3:
				deb.prints(batch["prediction"].shape)
				deb.prints(batch["labels"].shape)

			#self.prediction2old_labels_get(batch["prediction"])
			if im_reconstruct:
				self.reconstruct["idx"]+=idx*batch_size
				#deb.prints(idx)
				self.reconstruct=self.im_reconstruct(batch,self.reconstruct,self.reconstruct["idx"],batch_size,self.conf["patch"]["size"],self.conf["patch"]["overlap"],mask,label)
				#deb.prints(np.average(self.reconstruct["im"]))
			
			batch["correct_per_class"]=self.correct_per_class_get(batch["labels"],batch["prediction"])
			stats["correct_per_class"]+=batch["correct_per_class"]
			
			if self.debug>=3:
				deb.prints(batch["correct_per_class"])
				deb.prints(stats["correct_per_class"])
			
		stats["per_class_label_count"]=self.per_class_label_count_get(data["labels"])

		if im_reconstruct:
			deb.prints(np.unique(self.reconstruct["im"]))
			cv2.imwrite("reconstructed.png",self.reconstruct["im"])
		if self.debug>=1:
			deb.prints(data["labels"].shape)
			deb.prints(stats["correct_per_class"])
			deb.prints(stats["per_class_label_count"])
		#if utils.
		if self.squeeze_classes:
			stats["per_class_accuracy"],stats["average_accuracy"],stats["overall_accuracy"]=self.correct_per_class_average_get(stats["correct_per_class"], stats["per_class_label_count"])
		else:
			stats["per_class_accuracy"],stats["average_accuracy"],stats["overall_accuracy"]=self.correct_per_class_average_get(stats["correct_per_class"][1::], stats["per_class_label_count"][1::])
		if self.debug>=1:
			try: 
				deb.prints(stats["overall_accuracy"])
				deb.prints(stats["average_accuracy"])
			except:
				print("Deb prints error")
		if self.debug>=2:
			deb.prints(stats["per_class_accuracy"])
		return stats

	def reconstruct_init(self):
		reconstruct={}
		reconstruct["im"]=np.zeros(self.conf["im_size"])
		deb.prints(reconstruct["im"].shape)
		reconstruct["idx"]=0
		

		return reconstruct


		reconstruct["ids"]=self.reconstruct_ids_get(self.reconstruct["im"])

	def im_reconstruct(self,batch,reconstruct,batch_idx,batch_size,window=5,overlap=4,mask=None,label=None):

		patches_get={}
		h, w = reconstruct["im"].shape
		#mask=cv2.imread("TrainTestMask.png",0)
		gridx = range(0, w - window, window - overlap)
		gridx = np.hstack((gridx, w - window))

		gridy = range(0, h - window, window - overlap)
		gridy = np.hstack((gridy, h - window))
		#deb.prints(gridx.shape)
		#deb.prints(gridy.shape)
		
		counter=0
		patches_get["train_n"]=0
		patches_get["test_n"]=0
		patches_get["test_n_limited"]=0
		test_counter=0
		reconstruct["train"]={}
		reconstruct["test"]={}

		reconstruct["train"]["count"]=0
		reconstruct["test"]["count"]=0
		
		test_real_count=0
		

		count=0
		for i in range(len(gridx)):
			for j in range(len(gridy)):
				count+=1
				xx = gridx[i]
				yy = gridy[j]
				if count>=batch_idx and count<(batch_idx+batch_size):
					mask_patch = mask[yy: yy + window, xx: xx + window]
					label_patch = label[yy: yy + window, xx: xx + window]
					is_mask_from_train=mask_patch[self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]==1

					#if np.any(label_patch==0):
					#	continue
					if is_mask_from_train==True: # Train sample
						continue
						#im=self.ram_data["ims"][reconstruct["train"]["count"]]
						#reconstruct["im"][yy: yy + window, xx: xx + window]=mask_patch.copy()


					elif np.all(mask_patch==2): # Test sample
						test_counter+=1
						if test_counter>=self.conf["extract"]["test_skip"]:
							test_counter=0
							#print(batch_idx,batch_size,count)
							#print(batch["prediction"][test_counter])
							#print(np.argmax(batch["prediction"][test_counter]))
							#print(yy,xx)
							reconstruct["im"][yy+self.conf["patch"]["center_pixel"], xx+self.conf["patch"]["center_pixel"]]=np.argmax(batch["prediction"][test_real_count]).astype(np.uint8)
							test_real_count+=1
							
		return reconstruct

	def correct_per_class_get(self,target,prediction,debug=0):
		correct_per_class = np.zeros(self.n_classes).astype(np.float32)
		targets_int,predictions_int=self.targets_predictions_int_get(target,prediction)
		correct_all_classes = targets_int[targets_int == predictions_int]
		count_total = correct_all_classes.shape[0]
		
		if debug>=3: deb.prints(count_total)
		for clss in range(0,self.n_classes):
			correct_per_class[clss]=correct_all_classes[correct_all_classes==clss].shape[0]
		if debug>=3: deb.prints(correct_per_class)
		return correct_per_class
	def correct_per_class_average_get(self,correct_per_class,targets_label_count):
		correct_per_class_average=np.divide(correct_per_class, targets_label_count)
		accuracy_average=correct_per_class_average[~np.isnan(correct_per_class_average)]
		accuracy_average=accuracy_average[np.nonzero(accuracy_average)]
		accuracy_average=np.average(accuracy_average)
		overall_accuracy=np.sum(correct_per_class)/np.sum(targets_label_count)# Don't take backnd (label 0) into account for overall accuracy
		
		return correct_per_class_average, accuracy_average, overall_accuracy



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
		with tf.name_scope('init_definitions'):
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
			self.global_step = tf.placeholder(tf.int32, name='global_step')
			self.training = tf.placeholder(tf.bool, name='training')
			self.data,self.target=self.placeholder_init(self.timesteps,self.shape,self.channels,self.n_classes)
		with tf.name_scope('model_graph'):
			self.logits, self.prediction = self.model_graph_get(self.data)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.name, var)
		with tf.name_scope('optimization'):
			self.minimize,self.mistakes,self.error=self.loss_optimizer_set(self.target,self.prediction, self.logits)
			self.error_sum, self.saver, self.merged = self.tensorboard_saver_init(self.error)
			self.trainable_vars_print()
	def conv2d_block_get(self,graph_pipeline,filters,kernel=3,padding='same',training=True,layer_idx=0):
		graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, kernel, activation=None,padding=padding,name='conv2d_'+str(layer_idx))
		graph_pipeline=self.batchnorm(graph_pipeline,training=training,name='batchnorm'+str(layer_idx))
		graph_pipeline = tf.nn.relu(graph_pipeline,name='activation_'+str(layer_idx))
		
		return graph_pipeline
	def resnet_block_get(self,graph_pipeline,filters,kernel=3,padding='same',training=True,layer_idx=0):
		input_=tf.identity(graph_pipeline)
		resnet_idx=0

		layer_id='_'+str(layer_idx)+'_'+str(resnet_idx)
		graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, kernel, activation=None,padding=padding,name='resnet_conv2d'+layer_id)
		graph_pipeline=self.batchnorm(graph_pipeline,training=training,name='batchnorm'+layer_id)
		graph_pipeline = tf.nn.relu(graph_pipeline,name='activation'+layer_id)
	
		resnet_idx+=1
		layer_id='_'+str(layer_idx)+'_'+str(resnet_idx)
		graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, kernel, activation=None,padding=padding,name='resnet_conv2d'+layer_id)
		graph_pipeline=self.batchnorm(graph_pipeline,training=training,name='batchnorm'+layer_id)
		
		graph_pipeline=tf.add(input_,graph_pipeline)
		graph_pipeline = tf.nn.relu(graph_pipeline,name='activation'+layer_id)
		return graph_pipeline	

	
		

# ============================ NeuralNetSemantic takes image output ============================================= #

class NeuralNetSemantic(NeuralNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing NeuralNetSemantic instance")

	def placeholder_init(self,timesteps,shape,channels,n_classes):
		data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels], name='data')
		target = tf.placeholder(tf.float32, [None] + shape[0::], name='target')
		if self.debug: deb.prints(target.get_shape())
		return data,target

	def average_accuracy_get(self,target,prediction,debug=0):
		accuracy_average=0.5
		return accuracy_average


	def weighted_loss(self, logits, labels, num_classes, head=None):
		""" median-frequency re-weighting """
		with tf.name_scope('loss'):

			logits = tf.reshape(logits, (-1, num_classes))

			epsilon = tf.constant(value=1e-10)

			logits = logits + epsilon

			# consturct one-hot label array
			label_flat = tf.reshape(labels, (-1, 1))

			# should be [batch ,num_classes]
			labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

			softmax = tf.nn.softmax(logits)

			cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

			cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

			tf.add_to_collection('losses', cross_entropy_mean)

			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		return loss

	def cal_loss(self, logits, labels):
		loss_weight = np.array([
		  0,
		  0.6326076,
		  0,
		  0,
		  0.93579704,
		  1.,
		  0.82499779,
		  0.5,
		  0.74134727]) # class 0~11

		labels = tf.cast(labels, tf.int32)
		# return loss(logits, labels)
		return self.weighted_loss(logits, labels, num_classes=self.n_classes, head=loss_weight)

	def loss_optimizer_set(self,target,prediction, logits):

		tf.summary.image('prediction',tf.cast(tf.expand_dims(tf.multiply(prediction,20),axis=3),tf.uint8),max_outputs=10)

		tf.summary.image('target',tf.cast(tf.expand_dims(tf.multiply(target,20),axis=3),tf.uint8),max_outputs=10)
		
		target_int=tf.cast(target,tf.int32)
		deb.prints(target_int.get_shape())
		deb.prints(logits.get_shape())

		# Estimate loss from prediction and target
		#with tf.name_scope('cross_entropy'):
		#weights=tf.transpose(tf.constant([[0, 0.6326076 ,  0, 0, 0.93579704,  1.        ,  0.82499779,  0.5       , 0.74134727]]))

		#im_weights=tf.Variable(tf.zeros([32,32],dtype=tf.float32))
		#for clss in range(0,8):
		#	comparison = tf.equal( target_int, clss )
		#	print(clss)
		#	weight = tf.gather(weights, clss)
		#	print(weights[clss])
		#	print(weight.get_shape())
		#	im_weights = im_weights.assign_add( tf.where (comparison, tf.multiply(weight,tf.ones_like(im_weights)), im_weights) )

			#im_weights[target_int==clss]=weights[clss]


#graph_pipeline = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		#loss = self.cal_loss(logits, target_int)
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_int, logits=logits)

		deb.prints(loss.get_shape())
		cross_entropy = tf.reduce_mean(loss)
		deb.prints(cross_entropy.get_shape())

		#with tf.name_scope('learning_rate'):
		#learning_rate = tf.train.exponential_decay(0.1, self.global_step, 288, 0.96, staircase=True)
		
		#	learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step, opt.iter_epoch, opt.lr_decay, staircase=True)
		# Prepare the optimization function
		##optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
		#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		#optimizer = tf.train.AdamOptimizer()
		optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.0001)
		grads = optimizer.compute_gradients(cross_entropy)
		minimize = optimizer.apply_gradients(grads)
		# Save the grads with tf.summary.histogram
		for index, grad in enumerate(grads):
			tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

		error = tf.reduce_sum(-tf.cast(tf.abs(tf.subtract(tf.contrib.layers.flatten(tf.cast(prediction,tf.int64)),tf.contrib.layers.flatten(tf.cast(target,tf.int64)))), tf.float32))

		tf.summary.scalar('error',error)
		
		#minimize = optimizer.minimize(cross_entropy)
		#minimize = optimizer.minimize(error)
		
		prediction=tf.cast(prediction,tf.float32)
		# Distance L1
		mistakes=None
		return minimize, mistakes, error

	def unique_classes_print(self,data,memory_mode):
		count,unique=np.unique(data["labels"],return_counts=True)
		print("Train count,unique=",count,unique)
		pass

	def data_stats_get2(self,data,batch_size=1000):
		stats={}
		stats["average_accuracy"]=0
		stats["overall_accuracy"]=0
		return stats

	def batch_prediction_from_sess_get(self,ims):
		return self.sess.run(self.prediction,{self.data: ims, self.keep_prob: 1.0, self.training: False})

	def targets_predictions_int_get(self,target,prediction):
		return target.flatten(),prediction.flatten()

	def per_class_label_count_get(self,data_labels):
		per_class_label_count=np.zeros(self.n_classes)
		classes_unique,classes_count=np.unique(data_labels,return_counts=True)

		for clss,clss_count in zip(np.nditer(classes_unique),np.nditer(classes_count)):
			per_class_label_count[int(clss)]=clss_count
		deb.prints(per_class_label_count)
		return per_class_label_count
	def batchnorm(self,inputs,training=True,axis=3,name=None):
		return tf.layers.batch_normalization(inputs, axis=axis, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02), name=name)
	def conv2d_out_get(self,graph_pipeline,n_classes,kernel_size=3,padding='same',layer_idx=0):
		graph_pipeline=tf.layers.conv2d(graph_pipeline, n_classes, kernel_size, activation=None,padding=padding,name='conv2d_'+str(layer_idx))
		prediction = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		return graph_pipeline, prediction



# ============================ NeuralNet takes onehot image output ============================================= #
class NeuralNetOneHot(NeuralNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing NeuralNetOneHot instance")

	def placeholder_init(self,timesteps,shape,channels,n_classes):
		data = tf.placeholder(tf.float32, [None] +[timesteps] + shape + [channels], name='data')
		target = tf.placeholder(tf.float32, [None, n_classes], name='target')
		if self.debug: deb.prints(target.get_shape())
		return data,target

	def batch_prediction_from_sess_get(self,ims):
		return np.around(self.sess.run(self.prediction,{self.data: ims, self.keep_prob: 1.0, self.training: False}),decimals=2)

	def targets_predictions_int_get(self,target,prediction): 
		return np.argmax(target,axis=1),np.argmax(prediction,axis=1)

	def per_class_label_count_get(self,data_labels):
		return np.sum(data_labels,axis=0)

	def average_accuracy_get(self,target,prediction,debug=0):	
		
		correct_per_class=self.correct_per_class_get(target,prediction,debug=debug)
		targets_label_count = np.sum(target,axis=0)
		correct_per_class_average, accuracy_average = self.correct_per_class_average_get(correct_per_class, targets_label_count)
		return correct_per_class_average,correct_per_class,accuracy_average

	def loss_optimizer_set(self,target,prediction,logits=None):
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


	# From data stats get()
	def prediction2old_labels_get(self,predictions):
		# new_predictions=predictions.copy()
		# for i in range(len(classes)):
		# 	new_predictions[predictions == self.classes[i]] = self.new_labels2labels[self.classes[i]]
		# return new_predictions
		return predictions

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

	def test(self, args):
		
		self.sess = tf.Session()
		self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))

		print("Model restored.")
		data = self.data_load(self.conf,memory_mode=self.conf["memory_mode"])
		deb.prints(args.im_reconstruct)
		test_stats=self.data_stats_get(data["test"],im_reconstruct=args.im_reconstruct)

	def model_test_on_samples(self,dataset,sample_range=range(15,20)):

		print("train results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["train"]["ims"][sample_range], self.keep_prob: 1.0, self.training: False}),decimals=4))
		deb.prints(dataset["train"]["labels"][sample_range])
		
		print("test results")
		
		print(np.around(self.sess.run(self.prediction,{self.data: dataset["test"]["ims"][sample_range], self.keep_prob: 1.0, self.training: False}),decimals=4))
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
