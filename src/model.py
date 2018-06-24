
 

# from __future__ import division
# import os
# import math
# #import json
# import random
# #import pprint
# import time
# #import scipy.misc
# import numpy as np
# from time import gmtime, strftime
# import glob
# #from skimage.transform import resize
# #from sklearn import preprocessing as pre
# #import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# from random import shuffle
# #from tensorflow.contrib.rnn import ConvLSTMCell
# import glob
# import sys
# import pickle

# # Local
# import utils
# import deb
# import cv2
# #from cell import ConvGRUCell
# #from tf.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from model_base import *




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
class conv_lstm_semantic(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		graph_pipeline1=self.layer_lstm_get(data,filters=10,kernel=self.kernel,name='convlstm')
		
		if self.debug: deb.prints(graph_pipeline1.get_shape())
		#graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, strides=2, activation=None)
		self.layer_idx=0

		###graph_pipeline=self.resnet_block_get(graph_pipeline1,10,training=self.training,layer_idx=self.layer_idx,kernel=2)
		graph_pipeline=self.conv2d_block_get(graph_pipeline1,10,training=self.training,layer_idx=self.layer_idx,kernel=2)
		self.layer_idx+=1
		##if self.debug: deb.prints(graph_pipeline.get_shape())
		
		#graph_pipeline=self.conv2d_block_get(graph_pipeline,64,training=self.training,layer_idx=self.layer_idx)
		#self.layer_idx+=1
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.concat([graph_pipeline1,graph_pipeline],axis=3)

		graph_pipeline,prediction=self.conv2d_out_get(graph_pipeline,self.n_classes,kernel_size=1,layer_idx=self.layer_idx)
		self.layer_idx+=1
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return graph_pipeline,prediction

# ================================= Implements ConvLSTM ============================================== #
class conv_lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		graph_pipeline=self.layer_lstm_get(data,filters=8,kernel=self.kernel,name='convlstm')
		
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

# ================================= Implements ConvLSTM ============================================== #
class lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		if self.debug: deb.prints(data.get_shape())
		
		graph_pipeline = tf.reshape(data,[-1,self.timesteps,self.patch_len*self.patch_len*self.channels]) # Shape [None,32,32,6*6]
		if self.debug: deb.prints(graph_pipeline.get_shape())
		
		graph_pipeline=self.layer_flat_lstm_get(graph_pipeline,filters=128,kernel=self.kernel,name='convlstm')
		
		if self.debug: deb.prints(graph_pipeline.get_shape())
		#graph_pipeline=tf.layers.max_pooling2d(inputs=graph_pipeline, pool_size=[2, 2], strides=2)
		#graph_pipeline = tf.layers.conv2d(graph_pipeline, self.filters, self.kernel_size, strides=2, activation=None)
		
		#graph_pipeline = tf.contrib.layers.flatten(graph_pipeline)
		#if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.layers.dense(graph_pipeline, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(graph_pipeline.get_shape())
		graph_pipeline = tf.nn.dropout(graph_pipeline, self.keep_prob)
		
		graph_pipeline = tf.layers.dense(graph_pipeline, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(graph_pipeline.get_shape())
		return None,graph_pipeline


