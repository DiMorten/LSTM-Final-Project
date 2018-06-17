

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
from cell import ConvGRUCell
#from tf.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

np.set_printoptions(suppress=True)

# ===================================NeuralNet generic class ======================================================= #
# =================================== Might take onehot or image output ============================================= #
class NeuralNet(object):

	def __init__(self, sess=tf.Session(), batch_size=50, epoch=200, train_size=1e8,
						timesteps=utils.conf["t_len"], patch_len=32,
						kernel=[3,3], channels=7, filters=32, n_classes=6,
						checkpoint_dir='./checkpoint',log_dir=utils.conf["summaries_path"],data=None, conf=utils.conf, debug=1, \
						patience=5,squeeze_classes=True):
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
		
		self.early_stop["count"]=0
		self.early_stop["best"]=0
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
			self.early_stop["signal"]=self.early_stop_check(stats["overall_accuracy"],stats["average_accuracy"])
			if self.early_stop["signal"]:
				deb.prints(self.early_stop["best"])
				deb.prints(self.early_stop["best_aa"])
				break
			
			print("Average accuracy:{}, Overall accuracy:{}".format(stats["average_accuracy"],stats["overall_accuracy"]))
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch["idxs"],time.time() - start_time))

			print("Epoch - {}. Steps per epoch - {}".format(str(epoch),str(idx)))
		return self.early_stop
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

		
		if self.data_len_get(data["test"],memory_mode=self.conf["memory_mode"])>1000:
			data["sub_test"]=self.data_sub_data_get(data["test"],1000,memory_mode=self.conf["memory_mode"])
		else:
			data["sub_test"]=data["test"]
		#deb.prints(data["train"]["ims"].shape)
		deb.prints(data["train"]["labels"].shape)
		#deb.prints(data["test"]["ims"].shape)
		deb.prints(data["test"]["labels"].shape)

		self.train_batch_loop(args,batch,data)
		
	def early_stop_check(self,metric,metric_aux):
		if metric>self.early_stop["best"]:
			self.early_stop["best"]=metric
			self.early_stop["best_aa"]=metric_aux
			self.early_stop["count"]=0
		else:
			self.early_stop["count"]+=1
			if self.early_stop["count"]>=self.early_stop["patience"]:
				return True
			else:
				return False
		return False
	def data_stats_get(self,data,batch_size=1000):

		batch={}
		batch["idxs"] = data["n"] // batch_size
		if self.debug>=2:
			deb.prints(batch["idxs"])

		stats={"correct_per_class":np.zeros(self.n_classes).astype(np.float32)}
		stats["per_class_label_count"]=np.zeros(self.n_classes).astype(np.float32)
		
		for idx in range(0, batch["idxs"]):
			batch=self.batch_ims_labels_get(batch,data,batch_size,idx,memory_mode=self.conf["memory_mode"])
			batch["prediction"] = self.batch_prediction_from_sess_get(batch["ims"])
			
			if self.debug>=2:
				deb.prints(batch["prediction"].shape)
				deb.prints(batch["labels"].shape)

			#self.prediction2old_labels_get(batch["prediction"])
		   
			batch["correct_per_class"]=self.correct_per_class_get(batch["labels"],batch["prediction"])
			stats["correct_per_class"]+=batch["correct_per_class"]
			
			if self.debug>=2:
				deb.prints(batch["correct_per_class"])
				deb.prints(stats["correct_per_class"])
			
		stats["per_class_label_count"]=self.per_class_label_count_get(data["labels"])

		if self.debug>=2:
			deb.prints(data["labels"].shape)
			deb.prints(stats["correct_per_class"])
			deb.prints(stats["per_class_label_count"])
		#if utils.
		if self.squeeze_classes:
			stats["per_class_accuracy"],stats["average_accuracy"],stats["overall_accuracy"]=self.correct_per_class_average_get(stats["correct_per_class"], stats["per_class_label_count"])
		else:
			stats["per_class_accuracy"],stats["average_accuracy"],stats["overall_accuracy"]=self.correct_per_class_average_get(stats["correct_per_class"][1::], stats["per_class_label_count"][1::])
		if self.debug>=1: 
			deb.prints(stats["overall_accuracy"])
			deb.prints(stats["average_accuracy"])
		if self.debug>=2:
			deb.prints(stats["per_class_accuracy"])
		return stats
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
		self.keep_prob = tf.placeholder(tf.float32)
		self.global_step = tf.placeholder(tf.int32)
		self.training = tf.placeholder(tf.bool, name='training')
		self.data,self.target=self.placeholder_init(self.timesteps,self.shape,self.channels,self.n_classes)
		self.logits, self.prediction = self.model_graph_get(self.data)

		self.minimize,self.mistakes,self.error=self.loss_optimizer_set(self.target,self.prediction, self.logits)
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
		loss = self.cal_loss(logits, target_int)
		#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_int, logits=logits)

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
		

		error = tf.reduce_sum(-tf.cast(tf.abs(tf.subtract(tf.contrib.layers.flatten(tf.cast(prediction,tf.int64)),tf.contrib.layers.flatten(tf.cast(target,tf.int64)))), tf.float32))

		tf.summary.scalar('error',error)
		
		minimize = optimizer.minimize(cross_entropy)
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
	def batchnorm(self,inputs,training=True,axis=3):
		return tf.layers.batch_normalization(inputs, axis=axis, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

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

		test_stats=self.data_stats_get(data["test"])

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


# ================================= Implements U-Net ============================================== #

# Remote: python main.py -mm="ram" --debug=2 -po 27 -ts 5 -tnl 10000000 --batch_size=50 --filters=256 -pl=32 -m="smcnn_unet" -nap=16000
# Local: python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="smcnn_unet" -pl=32 -nap=16000
class UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=256
	def model_graph_get(self,data):
		graph_pipeline = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh,padding='same')
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), graph_pipeline
		return graph_pipeline, prediction
	def conv_block_get(self,graph_pipeline):
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=tf.nn.tanh,padding='same')
 
		graph_pipeline=self.batchnorm(graph_pipeline,training=True)


		return graph_pipeline
class SMCNN_UNet_large(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=10
	def model_graph_get(self,data):

		graph_pipeline = data
		graph_pipeline = tf.transpose(graph_pipeline, [0, 2, 3, 4, 1])
		graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(graph_pipeline.get_shape())

		self.filter_first=64
		#conv0=tf.layers.conv2d(graph_pipeline, self.filter_first, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv1=self.conv_block_get(graph_pipeline,self.filter_first*2)
		#conv1=self.conv_block_get(graph_pipeline,256)
		
		conv2=self.conv_block_get(conv1,self.filter_first*4)
		conv3=self.conv_block_get(conv2,self.filter_first*8)
		##kernel,bias=conv3.variables
		##tf.summary.histogram('conv3', kernel)
		#conv4=self.conv_block_get(conv3,self.filter_first*16)
		#up3=self.deconv_block_get(conv4,conv2,self.filter_first*8)
		
		#self.hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
		#graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		up4=self.deconv_block_get(conv3,conv2,self.filter_first*4)
		up5=self.deconv_block_get(up4,conv1,self.filter_first*2)
		up6=self.deconv_block_get(up5,graph_pipeline,self.filter_first)
		#kernel,bias=up6.variables
		#tf.summary.histogram('up6', kernel)
		
		graph_pipeline=self.out_block_get(up6,self.n_classes)
		#graph_pipeline = tf.layers.conv2d(up6, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), graph_pipeline
		return graph_pipeline, prediction

	def conv_2d(self,graph_pipeline,filters):
		graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, self.kernel_size, strides=(1,1), activation=None,padding='same')
		graph_pipeline=self.batchnorm(graph_pipeline,training=self.training,axis=3)
		
		#graph_pipeline=self.batchnorm(graph_pipeline,training=True)
		graph_pipeline = tf.nn.relu(graph_pipeline)
		return graph_pipeline
	def conv_block_get(self,graph_pipeline,filters):
		
		graph_pipeline = self.conv_2d(graph_pipeline,filters)
		graph_pipeline = self.conv_2d(graph_pipeline,filters)
		
		##graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, self.kernel_size, strides=(1,1), activation=tf.nn.relu,padding='same')
		##graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, self.kernel_size, strides=(1,1), activation=tf.nn.relu,padding='same')
		
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, self.kernel_size, strides=(2,2), activation=tf.nn.relu,padding='same')
		graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		deb.prints(graph_pipeline.get_shape())

		return graph_pipeline
	def deconv_2d(self,graph_pipeline,filters):
		graph_pipeline = tf.layers.conv2d_transpose(graph_pipeline, filters, self.kernel_size,strides=(2,2),activation=None,padding='same')
		graph_pipeline=self.batchnorm(graph_pipeline,training=self.training)
		graph_pipeline = tf.nn.relu(graph_pipeline)
		return graph_pipeline
	def deconv_block_get(self,graph_pipeline,layer,filters):
		##graph_pipeline = tf.layers.conv2d_transpose(graph_pipeline, filters, self.kernel_size,strides=(2,2),activation=tf.nn.relu,padding='same')
		graph_pipeline = self.deconv_2d(graph_pipeline,filters)
		graph_pipeline = tf.concat([graph_pipeline,layer],axis=3)
		graph_pipeline = self.conv_2d(graph_pipeline,filters)
		graph_pipeline = self.conv_2d(graph_pipeline,filters)
		
		deb.prints(graph_pipeline.get_shape())
		return graph_pipeline
	def out_block_get(self,graph_pipeline,filters):
		graph_pipeline = self.conv_2d(graph_pipeline,self.filter_first)
		graph_pipeline = tf.layers.conv2d(graph_pipeline, filters, (1,1), activation=None,padding='same')
		deb.prints(graph_pipeline.get_shape())
		return graph_pipeline


#================= Small SMCNN_Unet ========================================#
# Remote: python main.py -mm="ram" --debug=1 -po 1 -ts 1 -tnl 1000000 --batch_size=2000 --filters=256 -pl=5 -m="smcnn_unet" -nap=160000
class SMCNN_UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=256
	def model_graph_get(self,data):

		graph_pipeline = data
		graph_pipeline = tf.transpose(graph_pipeline, [0, 2, 3, 4, 1])
		graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(graph_pipeline.get_shape())

		graph_pipeline = self.conv_block_get(graph_pipeline)
		#graph_pipeline = self.conv_block_get(graph_pipeline)
		#graph_pipeline = self.conv_block_get(graph_pipeline)				

		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), graph_pipeline
		return graph_pipeline, prediction
	def conv_block_get(self,graph_pipeline):
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=None,padding='same')
		graph_pipeline=self.batchnorm(graph_pipeline,training=self.training)
		graph_pipeline = tf.nn.relu(graph_pipeline)
		
		return graph_pipeline

class SMCNN_UNet_small(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=10
	def model_graph_get(self,data):

		graph_pipeline = data
		graph_pipeline = tf.transpose(graph_pipeline, [0, 2, 3, 4, 1])
		graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(graph_pipeline.get_shape())

		conv1 = tf.layers.conv2d(graph_pipeline, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv2 = tf.layers.conv2d(conv1, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv3 = tf.layers.conv2d(conv2, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		#conv4 = tf.layers.conv2d(conv3, 256, self.kernel_size, activation=tf.nn.relu,padding='same')

		layer_in = tf.concat([conv4,conv2],axis=3)
		conv5 = tf.layers.conv2d(layer_in, 256, self.kernel_size, activation=tf.nn.relu,padding='same')

		layer_in = tf.concat([conv5,conv1],axis=3)
		graph_pipeline = tf.layers.conv2d(layer_in, self.n_classes, self.kernel_size, activation=None,padding='same')

		prediction = tf.argmax(graph_pipeline, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), graph_pipeline
		return graph_pipeline, prediction
	def conv_block_get(self,graph_pipeline):
		graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, activation=None,padding='same')
		graph_pipeline=self.batchnorm(graph_pipeline,training=self.training)
		graph_pipeline = tf.nn.relu(graph_pipeline)
		
		return graph_pipeline



# ================================= Implements ConvLSTM ============================================== #
class conv_lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		graph_pipeline=self.layer_lstm_get(data,filters=self.filters,kernel=self.kernel,name='convlstm')
		
		if self.debug: deb.prints(graph_pipeline.get_shape())
		#graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, strides=2, activation=None)
		
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return None,graph_pipeline


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
		return None,graph_pipeline
# ================================= Implements SMCNN ============================================== #
# Remote: python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"
class SMCNN(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		graph_pipeline = data
		graph_pipeline = tf.transpose(graph_pipeline, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]

		deb.prints(graph_pipeline.get_shape())

		graph_pipeline = tf.layers.conv2d(graph_pipeline, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		#graph_pipeline = tf.layers.dropout(graph_pipeline,rate=self.keep_prob,training=False,name='dropout')
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())


		return None,graph_pipeline

# ================================= Implements SMCNN ============================================== #
class SMCNNlstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		#graph_pipeline = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		graph_pipeline = data

		graph_pipeline=self.layer_lstm_get(data,filters=self.filters,kernel=[3,3],get_last=False,name="convlstm")

		graph_pipeline = tf.transpose(data, [0, 2, 3, 4, 1])
		graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(graph_pipeline.get_shape())

		#graph_pipeline = tf.layers.conv2d(graph_pipeline, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		#if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		#graph_pipeline = tf.layers.dropout(graph_pipeline,rate=self.keep_prob,training=False,name='dropout')
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())


		return None,graph_pipeline

class SMCNNlstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.activations=tf.nn.relu
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		graph_pipeline1 = data
		graph_pipeline1 = tf.transpose(graph_pipeline1, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		graph_pipeline1 = tf.reshape(graph_pipeline1,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]
		deb.prints(graph_pipeline1.get_shape())
		#graph_pipeline1 = tf.layers.conv2d(graph_pipeline1, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		if self.debug: deb.prints(graph_pipeline1.get_shape())
		
		graph_pipeline2=self.layer_lstm_get(data,filters=5,kernel=[3,3],get_last=True,name="convlstm")
		deb.prints(graph_pipeline2.get_shape())
		#graph_pipeline1 = tf.transpose(graph_pipeline1, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		#graph_pipeline1 = tf.reshape(graph_pipeline1,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]
		

		graph_pipeline=tf.concat([graph_pipeline1,graph_pipeline2],axis=3)

		deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.conv2d(graph_pipeline, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		
		
		graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		#graph_pipeline = tf.layers.dropout(graph_pipeline,rate=self.keep_prob,training=False,name='dropout')
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())


		return None,graph_pipeline
# ================================= Implements SMCNN ============================================== #
# Remote: python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"
class SMCNN_conv3d(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.kernel=[3,3,3]
		self.model_build()
		
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		graph_pipeline = data
		#graph_pipeline = tf.transpose(graph_pipeline, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		#graph_pipeline = tf.reshape(graph_pipeline,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]

		deb.prints(graph_pipeline.get_shape())

		#graph_pipeline = tf.layers.conv2d(graph_pipeline, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		graph_pipeline=tf.layers.conv3d(graph_pipeline,256,[3,3,3],padding='same',activation=tf.nn.tanh)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline=tf.layers.max_pooling3d(inputs=graph_pipeline, pool_size=[2,1,1], strides=[2,1,1],padding='same')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		#graph_pipeline = tf.layers.dropout(graph_pipeline,rate=self.keep_prob,training=False,name='dropout')
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())


		return None,graph_pipeline




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
		return None,graph_pipeline
