
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
#import json
import random
#import pprint
#import scipy.misc
import numpy as np
from time import gmtime, strftime
#from osgeo import gdal
import glob
#from skimage.transform import resize
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
import cv2
import pathlib
#from sklearn.feature_extraction.image import extract_patches_2d
#from skimage.util import view_as_windows
import sys
import pickle
# Local
import deb
import argparse

from skimage.util import view_as_windows

def mask_train_test_switch_from_path(path):
	mask=cv2.imread(path)
	out=mask_train_test_switch(mask)
	return out
def mask_train_test_switch(mask):
	out=mask.copy()
	out[mask==1]=2
	out[mask==2]=1
	return mask
def val_set_get(buffr,mode='stratified',validation_split=0.2):
	buffr['train']['idx']=range(buffr['train']['n'])
	clss_train_unique,clss_train_count=np.unique(buffr['train']['labels_int'],return_counts=True)
	deb.prints(clss_train_count)
	buffr['val']={'n':int(buffr['train']['n']*validation_split)}
	
	#===== CHOOSE VAL IDX
	#mode='stratified'
	if mode=='random':
		buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
		

		buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
		buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
	
	elif mode=='stratified':
		while True:
			buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
			buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
			buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
	
			clss_val_unique,clss_val_count=np.unique(buffr['val']['labels_int'],return_counts=True)
			
			if not np.array_equal(clss_train_unique,clss_val_unique):
				deb.prints(clss_train_unique)
				deb.prints(clss_val_unique)
				
				pass
			else:
				percentages=clss_val_count/clss_train_count
				deb.prints(percentages)
				#if np.any(percentages<0.1) or np.any(percentages>0.3):
				if np.any(percentages>0.23):
				
					pass
				else:
					break
	elif mode=='random_v2':
		while True:

			buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
			

			buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
			buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
			clss_val_unique,clss_val_count=np.unique(buffr['val']['labels_int'].argmax(axis=3),return_counts=True)
					
			deb.prints(clss_train_unique)
			deb.prints(clss_val_unique)

			deb.prints(clss_train_count)
			deb.prints(clss_val_count)

			clss_train_count_in_val=clss_train_count[np.isin(clss_train_unique,clss_val_unique)]
			percentages=clss_val_count/clss_train_count_in_val
			deb.prints(percentages)
			#if np.any(percentages<0.1) or np.any(percentages>0.3):
			if np.any(percentages>0.23):
				pass
			else:
				break				

	deb.prints(buffr['val']['idx'].shape)

	
	deb.prints(buffr['val']['ims'].shape)
	#deb.prints(data.patches['val']['labels_int'].shape)
	
	
	buffr['train']['ims']=np.delete(buffr['train']['ims'],buffr['val']['idx'],axis=0)
	buffr['train']['labels_int']=np.delete(buffr['train']['labels_int'],buffr['val']['idx'],axis=0)
	buffr['train']['n']=buffr['train']['ims'].shape[0]
	buffr['val']['n']=buffr['val']['ims'].shape[0]
	print("train",np.unique(buffr['train']['labels_int'],return_counts=True))
	print("val",np.unique(buffr['val']['labels_int'],return_counts=True))
	return buffr
def labels_onehot_get(labels,n_samples,class_n):
	out=np.zeros((n_samples,class_n))
	deb.prints(out.shape)
	deb.prints(labels.shape)
	out[np.arange(n_samples),labels.astype(np.int)]=1
	return out
def data_balance( data, samples_per_class,class_n,debug=1):
	fname=sys._getframe().f_code.co_name

	balance={}
	balance["unique"]={}
#	classes = range(0,self.conf["class_n"])
	classes,counts=np.unique(data["train"]["labels_int"],return_counts=True)
	print(classes,counts)
	num_total_samples=len(classes)*samples_per_class
	balance["out_labels"]=np.zeros(num_total_samples)
	deb.prints(num_total_samples)
	balance["out_data"]=np.zeros(num_total_samples)
	
	#balance["unique"]=dict(zip(unique, counts))
	#print(balance["unique"])
	k=0
	for clss in classes:
		deb.prints(clss,fname)
		balance["data"]=data["train"]["ims"][data["train"]["labels_int"]==clss]
		balance["labels_int"]=data["train"]["labels_int"][data["train"]["labels_int"]==clss]
		balance["num_samples"]=balance["data"].shape[0]
		if debug>=1: deb.prints(balance["data"].shape,fname)
		if debug>=2: 
			deb.prints(balance["labels_int"].shape,fname)
			deb.prints(np.unique(balance["labels_int"].shape),fname)
		if balance["num_samples"] > samples_per_class:
			replace=False
		else: 
			replace=True

		index = range(balance["labels_int"].shape[0])
		index = np.random.choice(index, samples_per_class, replace=replace)
		balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["labels_int"][index]
		balance["out_data"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["data"][index]

		k+=1
	idx = np.random.permutation(balance["out_labels"].shape[0])
	balance["out_data"] = balance["out_data"][idx]
	balance["out_labels"] = balance["out_labels"][idx]

	balance["labels"]=labels_onehot_get(balance["out_labels"],num_total_samples,class_n)
	#balance["labels"]=np.zeros((num_total_samples,self.conf["class_n"]))
	#balance["labels"][np.arange(num_total_samples),balance["out_labels"].astype(np.int)]=1
	if debug>=1: deb.prints(np.unique(balance["out_labels"],return_counts=True),fname)
	return balance["out_data"],balance["out_labels"],balance["labels"]

class DataForNet(object):
	def __init__(self,debug=1,patch_overlap=0,im_size=(948,1068),band_n=7,t_len=6,path="../data/",class_n=9,pc_mode="local", \
		patch_length=5,test_n_limit=1000,memory_mode="ram",flag_store=False,balance_samples_per_class=None,test_get_stride=None, \
		n_apriori=16000, squeeze_classes=False, data_dir='data',im_h=948,im_w=1068,id_first=1, \
		train_test_mask_name="TrainTestMask.tif",test_overlap_full=True,ram_store=True, \
		patches_save=False, test_folder=None, test_mode=False):
		deb.prints(patches_save)
		if test_mode=="True" or test_mode==True:
			self.test_mode=True
		else:
			self.test_mode=False
		self.patches_save=patches_save
		self.ram_store=ram_store
		self.conf={"band_n": band_n, "t_len":t_len, "path": path, "class_n":class_n, 'label':{}, 'seq':{}}
		self.conf["squeeze_classes"]=squeeze_classes
		self.conf["memory_mode"]=memory_mode #"ram" or "hdd"
		self.debug=debug
		self.test_n_limit=test_n_limit
		self.data_dir=data_dir
		self.conf["pc_mode"]=pc_mode
		self.conf['seq']['id_first']=id_first
		label_list=os.listdir(self.conf['path']+'labels/')
		deb.prints(self.conf['path']+'labels/')
		deb.prints(label_list)
		self.conf['seq']['id_list']=np.sort(np.array([int(x.partition('.')[0]) for x in label_list])) # Getting all label ids
		deb.prints(self.conf['seq']['id_list'])
		self.conf['seq']['id_max']=self.conf['seq']['id_list'][-1]
		#deb.prints(self.conf['seq']['id_list'])
		if self.debug>=1: deb.prints(self.conf['seq']['id_max'])
		if self.debug>=2: deb.prints(self.conf['seq']['id_list'])
		self.conf["label"]["last_name"]=str(self.conf['seq']['id_first']+t_len)+".tif"
		self.conf["label"]["last_dir"]=self.conf["path"]+"labels/"+self.conf["label"]["last_name"]
		self.conf["out_path"]=self.conf["path"]+"results/"
		self.conf["in_npy_path"]=self.conf["path"]+"in_np2/"
		self.conf["in_npy_path2"]=self.conf["path"]+"in_np2/"

		self.conf["in_rgb_path"]=self.conf["path"]+"in_rgb/"
		self.conf["in_labels_path"]=self.conf["path"]+"labels/"
		self.conf["patch"]={}
		self.conf["patch"]={"size":patch_length, "stride":5, "out_npy_path":self.conf["path"]+"patches_npy/"}
		self.conf["patch"]["ims_path"]=self.conf["patch"]["out_npy_path"]+"patches_all/"
		self.conf["patch"]["labels_path"]=self.conf["patch"]["out_npy_path"]+"labels_all/"
		self.conf['patch']['center_pixel']=int(np.around(self.conf["patch"]["size"]/2))
		deb.prints(self.conf['patch']['center_pixel'])
		self.conf["train"]={}
		self.conf["train"]["mask"]={}
		self.conf["train"]["mask"]["dir"]=self.conf["path"]+train_test_mask_name
		self.conf["train"]["ims_path"]=self.conf["path"]+"train_test/train/ims/"
		self.conf["train"]["labels_path"]=self.conf["path"]+"train_test/train/labels/"
		self.conf["test"]={}
		self.conf["test"]["ims_path"]=self.conf["path"]+"train_test/test/ims/"
		self.conf["test"]["labels_path"]=self.conf["path"]+"train_test/test/labels/"
		self.conf["test"]["overlap_full"]=test_overlap_full
		
		#self.conf["im_size"]=im_size
		self.conf["im_size"]=(im_h,im_w)
		deb.prints(self.conf["im_size"])
		deb.prints(self.conf["band_n"])		
		self.conf["im_3d_size"]=self.conf["im_size"]+(self.conf["band_n"],)
		self.conf["balanced"]={}

		self.conf["train"]["balanced_path"]=self.conf["path"]+"balanced/train/"
		self.conf["train"]["balanced_path_ims"]=self.conf["train"]["balanced_path"]+"ims/"
		self.conf["train"]["balanced_path_label"]=self.conf["train"]["balanced_path"]+"label/"
		self.conf["test"]["balanced_path"]=self.conf["path"]+"balanced/test/"
		self.conf["test"]["balanced_path_ims"]=self.conf["test"]["balanced_path"]+"ims/"
		self.conf["test"]["balanced_path_label"]=self.conf["test"]["balanced_path"]+"label/"

		self.conf["extract"]={}

		#self.conf["patch"]["overlap"]=26
		self.conf["patch"]["overlap"]=patch_overlap

		if self.conf["patch"]["overlap"]==26:
			self.conf["extract"]["test_skip"]=4
			self.conf["balanced"]["samples_per_class"]=150
		elif self.conf["patch"]["overlap"]==30:
			self.conf["extract"]["test_skip"]=10
			self.conf["balanced"]["samples_per_class"]=1500
		elif self.conf["patch"]["overlap"]==31:
			self.conf["extract"]["test_skip"]=24
			self.conf["balanced"]["samples_per_class"]=5000
		elif self.conf["patch"]["overlap"]==0:
			self.conf["extract"]["test_skip"]=0
			self.conf["balanced"]["samples_per_class"]=1000

		elif self.conf["patch"]["overlap"]>=2 or self.conf["patch"]["overlap"]<=3:
			self.conf["extract"]["test_skip"]=8
			self.conf["balanced"]["samples_per_class"]=1500
		elif self.conf["patch"]["overlap"]==4:
			self.conf["extract"]["test_skip"]=10
			self.conf["balanced"]["samples_per_class"]=5000
		if self.conf["pc_mode"]=="remote":
			self.conf["subdata"]={"flag":True,"n":3768}
		else:
			self.conf["subdata"]={"flag":True,"n":1000}
		#self.conf["subdata"]={"flag":True,"n":500}
		#self.conf["subdata"]={"flag":True,"n":1000}
		self.conf["summaries_path"]=self.conf["path"]+"summaries/"

		if balance_samples_per_class:
			self.conf["balanced"]["samples_per_class"]=balance_samples_per_class

		self.conf["extract"]["test_skip"]=test_get_stride

		deb.prints(self.conf["patch"]["overlap"])
		deb.prints(self.conf["extract"]["test_skip"])
		deb.prints(self.conf["balanced"]["samples_per_class"])
		

		pathlib.Path(self.conf["train"]["balanced_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["test"]["balanced_path"]).mkdir(parents=True, exist_ok=True) 

		self.conf["utils_main_mode"]=7
		self.conf["utils_flag_store"]=flag_store

		self.patch_shape=(self.conf["patch"]["size"],self.conf["patch"]["size"],self.conf["band_n"])
		self.label_shape=(self.conf["patch"]["size"],self.conf["patch"]["size"])
		
		self.conf["train"]["n_apriori"]=n_apriori
		self.conf["test"]["n_apriori"]=n_apriori
		#if self.conf["patch"]["overlap"]==0 and self.conf["patch"]["size"]==5:
		#	self.conf["train"]["n_apriori"]=3950
		#	if self.conf["extract"]["test_skip"]==0:
		#		self.conf["test"]["n_apriori"]=15587

		if self.test_n_limit<=self.conf["test"]["n_apriori"] and self.ram_store:
			self.conf["test"]["n_apriori"]=self.test_n_limit

		deb.prints(self.conf["train"]["n_apriori"])
		deb.prints(self.conf["test"]["n_apriori"])
		deb.prints(self.conf["class_n"])
		self.ram_data={"train":{},"test":{}}
		self.ram_data['test_folder']=test_folder
		if self.ram_store:
			print("HEEEERE")
			self.ram_data["train"]["ims"]=np.zeros((self.conf["train"]["n_apriori"],self.conf["t_len"])+self.patch_shape)
			self.ram_data["test"]["ims"]=np.zeros((self.conf["test"]["n_apriori"],self.conf["t_len"])+self.patch_shape)
		
		self.conf["label_type"]="one_hot"

		print(self.conf)

		#self.im_npy_get()
	def im_npy_get(self):
		pathlib.Path(self.conf["in_npy_path2"]).mkdir(parents=True, exist_ok=True) 

		#for i in range(1,10):
		for i in range(1,10):
			im_folder_name=glob.glob(self.conf["path"]+'im'+str(i)+'*')[0]+"/"
			#im=cv2.imread(im_name)
			if self.debug>=3: deb.prints(im_folder_name)
			im_band_path_list=[glob.glob(im_folder_name+"*band1*.tif"), \
			glob.glob(im_folder_name+"*band2*.tif"), \
			glob.glob(im_folder_name+"*band3*.tif"), \
			glob.glob(im_folder_name+"*band4*.tif"), \
			glob.glob(im_folder_name+"*band5*.tif"), \
			glob.glob(im_folder_name+"*band7*.tif")]
			if self.debug>=3: deb.prints(im_band_path_list)

			im_all_bands=np.zeros(self.conf["im_size"] + (6,))
			counter=0
			for im_band_path in im_band_path_list:
				deb.prints(im_band_path[0])
				im_all_bands[:,:,counter]=cv2.imread(im_band_path[0],-1)
				if self.debug>=4: deb.prints(im.shape)
				if self.debug>=4: deb.prints(im.dtype)
				counter+=1
			ndvi = np.expand_dims((im_all_bands[:,:,3] - im_all_bands[:,:,2])/(im_all_bands[:,:,3] + im_all_bands[:,:,2]),axis=2)
			deb.prints(ndvi.shape)
			deb.prints(im_all_bands.shape)
			im_all_bands = np.concatenate((im_all_bands, ndvi),axis=2)
			deb.prints(im_all_bands.shape)
			np.save(self.conf["in_npy_path2"]+"im"+str(i)+".npy",im_all_bands.astype(np.float64))



			#im_name=im_name[-14:-4]
			#im_names.append(im_name)
			#print(im_name)

	def patches_load(self,label_type="one_hot"):
		fname=sys._getframe().f_code.co_name
		print('[@im_patches_npy_multitemporal_from_npy_store2]')
		#add_id=self.conf['seq']['id_max']-self.conf["t_len"] # Future: deduce this 9 from data
		##add_id=self.conf['seq']['id_first']-1 # Future: deduce this 9 from data
		
		
		# This is where I want seq2 to be
		#foldername='/mnt/Data/Jorge/tf_patches/seq2_overlap4_masked_norm/patch_npy/'
		
		# This is seq1
		#foldername='/mnt/Data/Jorge/tf_patches/seq1_overlap4_masked_norm/patch_npy/'
		
		
		# This will be hannover
		#foldername='/mnt/Data/Jorge/tf_patches/hannover_overlap4_masked_norm/patch_npy/'
		

		# This is seq1 complete 5x5
		#foldername='/mnt/Data/Jorge/tf_patches/seq1_overlap4_masked_norm_complete/patch_npy/'
		
		# This is seq1 7x7
		#foldername='/mnt/Data/Jorge/tf_patches/seq1_overlap6_7x7_masked_norm/patch_npy/'
		#foldername=self.conf["path"]+'patch_npy/'

		self.patches_create=True
		#self.ram_store=False
		if self.patches_create==True:
			foldername=self.conf["path"]+'patch_npy/'
			add_id=0
			if self.debug>=1: 
				deb.prints(add_id)
				deb.prints(self.conf['seq']['id_max'])
				deb.prints(self.conf["t_len"])
			pathlib.Path(self.conf["patch"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
			pathlib.Path(self.conf["patch"]["labels_path"]).mkdir(parents=True, exist_ok=True) 
			patches_all=np.zeros((58,65,self.conf["t_len"])+self.patch_shape)
			label_patches_all=np.zeros((58,65,self.conf["t_len"])+self.label_shape)
		

			#========================== GET IMAGE FILENAMES =================================#
			im_filenames=self.im_filenames_get()



			patch={}
			deb.prints(self.conf["train"]["mask"]["dir"])
			patch["train_mask"]=cv2.imread(self.conf["train"]["mask"]["dir"],-1).astype(np.uint8)

			deb.prints((self.conf["t_len"],)+self.patch_shape)

			#patch["values"]=np.zeros((self.conf["t_len"],)+patch_shape)
			# Init full ims and labels arrays
			patch["full_ims"]=np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"])
			patch["full_label_ims"]=np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"][0:2])

			#for t_step in range(0,self.conf["t_len"]):
			


			#=======================LOAD, NORMALIZE AND MASK FULL IMAGES ================#
			patch=self.im_load(patch,im_filenames,add_id)

			deb.prints(im_filenames)

			patch["full_ims"][patch["full_ims"]>1]=1
			patch["full_ims"]=self.im_seq_normalize(patch["full_ims"])
			deb.prints(np.min(patch["full_ims"]))
			deb.prints(np.max(patch["full_ims"]))
			deb.prints(patch["full_ims"].dtype)
			#self.full_ims_train,self.full_ims_test=self.im_seq_mask(patch["full_ims"],patch["train_mask"])
			self.full_ims_train=patch["full_ims"].copy()
			self.full_ims_test=patch['full_ims'].copy()

			patch["full_ims"]=None
			deb.prints(self.full_ims_train.dtype)
			self.full_label_train,self.full_label_test=self.label_seq_mask(patch["full_label_ims"][self.conf['t_len']-1],patch["train_mask"]) 
			deb.prints(self.full_label_train.dtype)
			#self.label_id=self.conf["seq"]["id_first"]+self.conf['t_len']-2 # Less 1 for python idx, less 1 for id_first starts at 1 
		 
			unique,count=np.unique(self.full_label_train,return_counts=True) 
			print("Train masked unique/count",unique,count) 
			unique,count=np.unique(self.full_label_test,return_counts=True) 
			print("Test masked unique/count",unique,count) 
			# Load train mask
			#self.conf["patch"]["overlap"]=26

			pathlib.Path(self.conf["train"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
			pathlib.Path(self.conf["train"]["labels_path"]).mkdir(parents=True, exist_ok=True) 
			pathlib.Path(self.conf["test"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
			pathlib.Path(self.conf["test"]["labels_path"]).mkdir(parents=True, exist_ok=True) 


			deb.prints(patch["train_mask"])
			deb.prints(self.conf["train"]["mask"]["dir"])
		

			#========================== BEGIN PATCH EXTRACTION ============================#
			#view_as_windows_flag=False
			
			view_as_windows_flag="3"
			self.ram_store=True # This should be removed for normal 
			set_path_test=False
			path_save=self.conf['path']+'buffer/'
			path_train=path_save+'train/'
			path_test=self.ram_data['test_folder']
			if set_path_test==True:

				
				
				path_test=path_save+'test_batched_7/'
				path_test=path_save+'test_batched_5/'
				path_test=path_save+'test_batched_15/'

				self.ram_data['test_folder']=path_test
			#path_test=path_save+'test_batched_19/'
			
			
			
			#path_test=path_save+'test_batched/'
			#path_test=path_save+'test_batched_11/'
			
			
			pathlib.Path(path_train).mkdir(parents=True, exist_ok=True)
			pathlib.Path(path_test).mkdir(parents=True, exist_ok=True)			
			
			if view_as_windows_flag==True:
				self.conf["train"]["n"],self.conf["test"]["n"]=self.patches_multitemporal_get2(patch["full_ims"],patch["full_label_ims"], \
					self.conf["patch"]["size"],self.conf["patch"]["overlap"],mask=patch["train_mask"],path_train=self.conf["train"], \
					path_test=self.conf["test"],patches_save=self.patches_save,label_type=label_type,memory_mode=self.conf["memory_mode"],\
					foldername=foldername)
			elif view_as_windows_flag==False:

				#path_save='/mnt/Data/Jorge/tf_patches/atomic/seq1_11x11/'
				#path_save='/media/lvc/80D0DD48D0DD4556/Jorge/atomic/seq1_11x11'
				
				
				self.conf["train"]["n"],self.conf["test"]["n"]=self.patches_multitemporal_get(patch["full_ims"],patch["full_label_ims"], \
					self.conf["patch"]["size"],self.conf["patch"]["overlap"],mask=patch["train_mask"],path_train=self.conf["train"], \
					path_test=self.conf["test"],patches_save=self.patches_save,label_type=label_type,memory_mode=self.conf["memory_mode"])
				deb.prints(self.conf["test"]["overlap_full"])
				#print(self.conf["test"]["overlap_full"]==True)
				#print(self.conf["test"]["overlap_full"]=="True")
				if self.conf["test"]["overlap_full"]=="True" or self.conf["test"]["overlap_full"]==True:
					# Make test with overlap full
					_,self.conf["test"]["n"]=self.patches_multitemporal_get(patch["full_ims"],patch["full_label_ims"], \
						self.conf["patch"]["size"],self.conf["patch"]["size"]-1,mask=patch["train_mask"],path_train=self.conf["train"], \
						path_test=self.conf["test"],patches_save=self.patches_save,label_type=label_type,memory_mode=self.conf["memory_mode"],test_only=True)
			elif view_as_windows_flag=="3":
				self.conf["train"]["n"],self.conf["test"]["n"]=self.patches_multitemporal_get3(patch["full_ims"],patch["full_label_ims"], \
					self.conf["patch"]["size"],self.conf["patch"]["overlap"],mask=patch["train_mask"],path_train=path_train, \
					path_test=path_test,patches_save=self.patches_save,label_type=label_type,memory_mode=self.conf["memory_mode"])

			patch["full_ims"]=None
			deb.prints(self.ram_data['test']['ims'].shape)
			deb.prints(self.ram_data['test']['labels_int'].shape)
			deb.prints(self.ram_data['test']['ims'].dtype)
			deb.prints(self.ram_data['test']['labels_int'].dtype)
		
		
			deb.prints(self.conf["test"]["n"])

			deb.prints(self.ram_data['train']['ims'].shape)
			#self.val_set_get(mode='stratified',validation_split=0.15)
			deb.prints(self.ram_data['train']['ims'].shape)
			deb.prints(self.ram_data['val']['ims'].shape)
			
			self.ram_data_store=False
			if self.ram_data_store:
				
				pathlib.Path(self.conf["path"]+foldername).mkdir(parents=True, exist_ok=True) 

				#np.save(self.conf["path"]+foldername+"ram_data.npy",self.ram_data)
				np.save(self.conf["path"]+foldername+"train_ims.npy",self.ram_data["train"]['ims'])
				np.save(self.conf["path"]+foldername+"train_labels_int.npy",self.ram_data['train']['labels_int'])
				np.save(self.conf["path"]+foldername+"train_n.npy",self.ram_data['train']['n'])
				
				np.save(self.conf["path"]+foldername+"test_ims.npy",self.ram_data["test"]['ims'])
				np.save(self.conf["path"]+foldername+"test_labels_int.npy",self.ram_data['test']['labels_int'])
				np.save(self.conf["path"]+foldername+"test_n.npy",self.ram_data['test']['n'])
				np.save(self.conf["path"]+foldername+"val_ims.npy",self.ram_data["val"]['ims'])
				np.save(self.conf["path"]+foldername+"val_labels_int.npy",self.ram_data['val']['labels_int'])
				np.save(self.conf["path"]+foldername+"val_n.npy",self.ram_data['val']['n'])
				
		else:
			print("Loading dataset...")
			#self.ram_store=False
			if self.ram_store:

				foldername=self.conf['path']

				self.ram_data={'train':{},'test':{}}
				self.ram_data['train']['ims']=np.load(foldername+"train_ims.npy")
				deb.prints(self.ram_data['train']['ims'].shape)
				self.ram_data['train']['labels_int']=np.load(foldername+"train_labels_int.npy")
				deb.prints(np.unique(self.ram_data['train']['labels_int']))
				
				deb.prints(self.ram_data['train']['labels_int'].shape)
				
				self.ram_data['train']['n']=self.ram_data['train']['ims'].shape[0]
				
				self.ram_data['val']={}
				self.ram_data['val']['ims']=np.load(foldername+"val_ims.npy")
				deb.prints(self.ram_data['val']['ims'].shape)
				
				self.ram_data['val']['labels_int']=np.load(foldername+"val_labels_int.npy")
				self.ram_data['val']['n']=self.ram_data['val']['ims'].shape[0]


				"""
				self.ram_data['test']['ims']=np.load(foldername+"test_ims.npy")
				deb.prints(self.ram_data['test']['ims'].shape)
				
				self.ram_data['test']['labels_int']=np.load(foldername+"test_labels_int.npy")
				self.ram_data['test']['n']=np.load(foldername+"test_n.npy")
				"""

		# ================== PATCHES ARE STORED IN self.ram_data ========================#

	def im_filenames_get(self):
		im_names=[]
		#for i in range(1,10):
		for i in range(self.conf['seq']['id_first'],self.conf['t_len']+self.conf['seq']['id_first']):
			im_name=glob.glob(self.conf["in_npy_path"]+'im'+str(i)+'.npy')[0]
			print(im_name)
			#im_name=im_name[-14:-4]
			im_name='im'+str(i)
			im_names.append(im_name)
			print(im_name)
		print(im_names)
		return im_names
		self.im_patches_npy_multitemporal_from_npy_store2(im_names,label_type)

	def im_load(self,patch,names,add_id):
		fname=sys._getframe().f_code.co_name
		for t_step in range(0,self.conf["t_len"]):	
			print(t_step,add_id)
			deb.prints(self.conf["in_npy_path"]+names[t_step]+".npy")
			patch["full_ims"][t_step] = np.load(self.conf["in_npy_path"]+names[t_step]+".npy")
			deb.prints(np.average(patch["full_ims"][t_step]))
			deb.prints(np.max(patch["full_ims"][t_step]))
			deb.prints(np.min(patch["full_ims"][t_step]))
			
			#deb.prints(patch["full_ims"][t_step].dtype)
			patch["full_label_ims"][t_step] = cv2.imread(self.conf["path"]+"labels/"+names[t_step][2:]+".tif",0)
			print("Label path", self.conf["path"]+"labels/"+names[t_step][2:]+".tif")
			#for band in range(0,self.conf["band_n"]):
			#	patch["full_ims_train"][t_step,:,:,band][patch["train_mask"]!=1]=-1
			# Do the masking here. Do we have the train labels?
		#patch["full_ims"]=patch["full_ims"].astype(np.float32)
		#patch["full_label_ims"]=patch["full_label_ims"].astype(np.uint8)
		
		deb.prints(patch["full_ims"].shape,fname)
		deb.prints(patch["full_label_ims"].shape,fname)
		return patch
		

	def patches_multitemporal_get2(self,img,label,window,overlap,mask,path_train,path_test,patches_save=True, \
		label_type="one_hot",memory_mode="hdd",test_only=False, ram_store=True,foldername="folder"):
		fname=sys._getframe().f_code.co_name
		patches={}
		masks={}
		deb.prints(img.shape,fname)
		step=window-overlap
		deb.prints(step,fname)
		

		# ============== GET TRAIN / TEST MASK ==============#

		masks['train']=mask.copy()
		masks['train'][masks['train']==2]=0
		masks['test']=mask.copy()
		masks['test'][masks['test']==1]=0
		masks['test'][masks['test']==2]=1

		print(label[self.conf['t_len']-1].dtype,label[self.conf['t_len']-1].shape)
		print(masks['train'].dtype,masks['train'].shape)
		masks['label_train']=cv2.bitwise_and(label[self.conf['t_len']-1],label[self.conf['t_len']-1],mask=masks['train'])
		masks['label_test']=cv2.bitwise_and(label[self.conf['t_len']-1],label[self.conf['t_len']-1],mask=masks['test'])
		

		patches=self.im_label_view_as_windows(img,label[self.conf['t_len']-1],mask,window,overlap,step,patches,foldername=foldername)

		patches,count=self.im_label_train_test_split(patches,path_train,path_test)

		self.ram_data['train']['ims']=patches['train'][0:count['train']]
		self.ram_data['train']['labels_int']=patches['label_train'][0:count['train']]
		self.ram_data['train']['n']=count['train']
		self.ram_data['test']['ims']=patches['test'][0:count['test']]
		self.ram_data['test']['labels_int']=patches['label_test'][0:count['test']]
		self.ram_data['test']['n']=count['test']
		return count['train'],count['test']

	def im_label_view_as_windows(self,img,label,mask,window,overlap,step,patches,foldername):
		for t_step in range(0,self.conf["t_len"]):
			out = np.squeeze(view_as_windows(img[t_step,:,:,:], (window,window,self.conf["band_n"]), step=step))
			out = np.reshape(out,(out.shape[0]*out.shape[1],))
			deb.prints(out.shape)
			patches['all_n']=out.shape[0]*out.shape[1]
		
		out=None
		patches['label_all']=np.zeros((patches['all_n'],window,window,2))	
		out=None
		patches['all']=np.zeros((patches['all_n'],self.conf['t_len'],window,window,self.conf['band_n']))	
		
		for t_step in range(0,self.conf["t_len"]):
			out = np.squeeze(view_as_windows(img[t_step,:,:,:], (window,window,self.conf["band_n"]), step=step))
			patches['all'][:,t_step,:,:,:] = np.reshape(out,(out.shape[0]*out.shape[1],out.shape[2],out.shape[3],out.shape[4]))
			print("Taking image windows...")
		out = np.squeeze(view_as_windows(label, (window,window), step=step))	
		patches['label_all']=np.reshape(out,(out.shape[0]*out.shape[1],out.shape[2],out.shape[3]))

		out = np.squeeze(view_as_windows(mask, (window,window), step=step))	
		patches['mask']=np.reshape(out,(out.shape[0]*out.shape[1],out.shape[2],out.shape[3]))
		out=None
		deb.prints(patches['all'].shape)
		deb.prints(patches['mask'].shape)
		deb.prints(patches['label_all'].shape)
		
		# Flatten
		patches['all']=np.reshape(patches['all'],(patches['all_n'],self.conf['t_len']*window*window*self.conf['band_n']))
		patches['mask']=np.reshape(patches['mask'],(patches['all_n'],window*window))
		patches['label_all']=np.reshape(patches['label_all'],(patches['all_n'],window*window))
		
		deb.prints(patches['all'].shape)
		deb.prints(patches['mask'].shape)
		deb.prints(patches['label_all'].shape)
		
		# Train ims
		tmp=patches['all'][np.any(patches['mask']==1,axis=1)]
		tmp=np.reshape(tmp,(tmp.shape[0],self.conf['t_len'],window,window,self.conf['band_n']))
		deb.prints(tmp.shape)
		np.save(self.conf["path"]+foldername+"train_ims.npy",tmp)
		
		# Train labels (int)
		tmp=patches['label_all'][np.any(patches['mask']==1,axis=1)]
		tmp=np.reshape(tmp,(tmp.shape[0],window,window))
		deb.prints(tmp.shape)
		np.save(self.conf["path"]+foldername+"train_labels_int.npy",tmp)
		
		np.save(self.conf["path"]+foldername+"train_n.npy",tmp.shape[0])
				

		# Test ims
		tmp=patches['all'][np.any(patches['mask']==2,axis=1)]
		tmp=np.reshape(tmp,(tmp.shape[0],self.conf['t_len'],window,window,self.conf['band_n']))
		deb.prints(tmp.shape)
		np.save(self.conf["path"]+foldername+"test_ims.npy",tmp)
		

		
		# Test labels (int)
		tmp=patches['label_all'][np.any(patches['mask']==2,axis=1)]
		tmp=np.reshape(tmp,(tmp.shape[0],window,window))
		deb.prints(tmp.shape)
		np.save(self.conf["path"]+foldername+"test_labels_int.npy",tmp)

		np.save(self.conf["path"]+foldername+"test_n.npy",tmp.shape[0])


		return patches	

	def im_label_train_test_split(self,patches,path_train,path_test):
		print("[@im_label_train_test_split]")
		count={'train':0, 'test':0}

		patches['train']=np.zeros_like(patches['all'])
		patches['test']=np.zeros_like(patches['all'])
		
		patches['label_train']=np.zeros_like(patches['label_all'])
		patches['label_test']=np.zeros_like(patches['label_all'])
		
		for i in range(0,patches['all'].shape[0]):

			no_zero=True
			if np.all(patches['label_all'][i]==0) and no_zero==True:
				continue
			is_mask_from_train=self.is_mask_from_train(patches['mask'][i])
			if is_mask_from_train:
				mask_train=patches['mask'][i].copy()
				mask_train[mask_train==2]=0
				patches['train'][count['train']]=patches['all'][i]
				patches['label_train'][count['train']]=cv2.bitwise_and(patches['label_all'][i],patches['label_all'][i],mask=mask_train)
				


				if self.patches_save==True or self.patches_save=="True":
					#if self.conf["squeeze_classes"]==True or self.conf["squeeze_classes"]=="True":
					#	label_patch_parsed=self.labels_unused_classes_eliminate_prior(label_patch[self.conf["t_len"]-1])
					#else:
					#	label_patch_parsed=label_patch[self.conf["t_len"]-1].copy()
					#print("HEERERER")
					np.save(path_train["ims_path"]+"patch_"+str(count['train'])+".npy",patches['train'][count['train']])
					np.save(path_train["labels_path"]+"patch_"+str(count['train'])+".npy",patches['label_train'][count['train']])


				count['train']+=1
			is_mask_from_test=self.is_mask_from_test(patches['mask'][i])
			if is_mask_from_test:
				mask_test=patches['mask'][i].copy()
				mask_test[mask_test==1]=0
				mask_test[mask_test==2]=1
				patches['test'][count['test']]=patches['all'][i]
				patches['label_test'][count['test']]=cv2.bitwise_and(patches['label_all'][i],patches['label_all'][i],mask=mask_test)
				
				if self.patches_save==True or self.patches_save=="True":
					##if self.conf["squeeze_classes"]==True or self.conf["squeeze_classes"]=="True":
					##	label_patch_parsed=self.labels_unused_classes_eliminate_prior(label_patch[self.conf["t_len"]-1])
					##else:
					##	label_patch_parsed=label_patch[self.conf["t_len"]-1].copy()
					np.save(path_test["ims_path"]+"patch_"+str(count['test'])+".npy",patches['test'][count['test']])
					np.save(path_test["labels_path"]+"patch_"+str(count['test'])+".npy",patches['label_test'][count['test']])

				count['test']+=1

		deb.prints(count)



		return patches,count
	def patches_multitemporal_get(self,img,label,window,overlap,mask,path_train,path_test,patches_save=True, \
		label_type="one_hot",memory_mode="hdd",test_only=False, ram_store=True):

		fname=sys._getframe().f_code.co_name

		deb.prints(window,fname)
		deb.prints(overlap,fname)
		print("STARTED PATCH EXTRACTION")
		#window= 256
		#overlap= 200
		patches_get={}
		t_steps, h, w, channels = img.shape
		mask_train=np.zeros((h,w))
		mask_test=np.zeros((h,w))
		
		gridx = range(0, w - window, window - overlap)
		gridx = np.hstack((gridx, w - window))

		gridy = range(0, h - window, window - overlap)
		gridy = np.hstack((gridy, h - window))
		deb.prints(gridx.shape)
		deb.prints(gridy.shape)
		
		counter=0
		patches_get["train_n"]=0
		patches_get["test_n"]=0
		patches_get["test_n_limited"]=0
		test_counter=0
		test_real_count=0
		

		deb.prints(self.patches_save)

		# This value checks which patches were taken into accoutn
		mask_covered_areas=np.zeros((h,w))

		if self.ram_store==False:
			self.unused_classes_elimination_configure(label[self.conf['t_len']-1])
			self.train_labels=[]
			self.test_labels=[]
		#======================== START IMG LOOP ==================================#
		for i in range(len(gridx)):
			for j in range(len(gridy)):
				counter=counter+1
				if counter % 10000000 == 0:
					deb.prints(counter,fname)
				xx = gridx[i]
				yy = gridy[j]
				#patch_clouds=Bclouds[yy: yy + window, xx: xx + window]
				patch = img[:,yy: yy + window, xx: xx + window,:]
				label_patch = label[:,yy: yy + window, xx: xx + window]
				mask_patch = mask[yy: yy + window, xx: xx + window].astype(np.float64)
				
				patch_train = self.full_ims_train[:,yy: yy + window, xx: xx + window,:]
				patch_test = self.full_ims_test[:,yy: yy + window, xx: xx + window,:]
				
				is_mask_from_train=self.is_mask_from_train(mask_patch,label_patch[self.conf["t_len"]-1])
				
				no_zero=True
				#if np.count_nonzero(label_patch[label_patch==0])>=1000 and no_zero==True:				
				
				if np.all(label_patch==0) and no_zero==True:
					continue
				#deb.prints(is_mask_from_train)
				#elif np.all(mask_patch==1): # Train sample
				
				#=======================IS PATCH FROM TRAIN =================================#
				if is_mask_from_train==True: # Train sample
					patch = patch_train.copy()
					#deb.prints("train")
					mask_train_areas=mask_patch.copy()
					mask_train_areas[mask_train_areas==2]=0 # Remove test from this patch
					mask_train[yy: yy + window, xx: xx + window]=mask_train_areas.astype(np.uint8)*255
					
					"""
					for t_step in range(0,self.conf["t_len"]):
						label_patch[t_step]=cv2.bitwise_and(label_patch[t_step],label_patch[t_step],mask=mask_train_areas.astype(np.uint8))
					center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
					if center_label==0:
						print("A2")
					
					label_patch[self.conf['t_len']-1]=self.full_label_train[yy: yy + window, xx: xx + window]
					center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
					if center_label==0:
						print("A3")
					"""
					if self.conf["memory_mode"]=="ram" and self.ram_store==True:
						if not test_only:
							self.ram_data["train"]=self.in_label_ram_store(self.ram_data["train"],patch,label_patch,data_idx=patches_get["train_n"],label_type=label_type,name="train")
					
					#print("herherher")
					if self.patches_save==True or self.patches_save=="True":
						label_center=label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]
						if self.conf["squeeze_classes"]==True or self.conf["squeeze_classes"]=="True":
							label_parsed=self.labels_unused_classes_eliminate_prior(label_center)
						else:
							label_parsed=label_center.copy()
						#print("HEERERER")
						np.save(path_train+"patch_"+str(patches_get["train_n"])+"_"+str(i)+"_"+str(j)+".npy",patch)
						self.train_labels.append(label_parsed)
						
					patches_get["train_n"]+=1	
				is_mask_from_test=self.is_mask_from_test(mask_patch,label_patch[self.conf["t_len"]-1])
				
				#============================ IS PATCH FROM TEST ===============================#
				if is_mask_from_test==True: # Test sample
					patch=patch_test.copy()
					#deb.prints("test")
					test_counter+=1
					
					#if np.random.rand(1)[0]>=0.7:
					
					patches_get["test_n"]+=1
					#if patches_get["test_n"]<=self.test_n_limit:
					if True:
						patches_get["test_n_limited"]+=1					
						##if test_counter>=self.conf["extract"]["test_skip"]:
						mask_test,label_patch=self.mask_test_update(mask_test,yy,xx,window,label_patch,mask_patch)
							##deb.prints(np.average(mask_test))
							#mask_test[yy: yy + window, xx: xx + window]=255
							#mask_test[int(yy + window/2), int(xx + window/2)]=255
						##	test_counter=0
							
							
					
							#if self.conf["memory_mode"]=="hdd":
							#deb.prints(self.ram_store)
						if self.conf["memory_mode"]=="ram" and self.ram_store==True:
							self.ram_data["test"]=self.in_label_ram_store(self.ram_data["test"],patch,label_patch,data_idx=test_real_count,label_type=label_type,name="test")
						if self.patches_save==True or self.patches_save=="True":
							
							label_center=label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]
							if self.conf["squeeze_classes"]==True or self.conf["squeeze_classes"]=="True":
								label_parsed=self.labels_unused_classes_eliminate_prior(label_center)
							else:
								label_parsed=label_center.copy()
							np.save(path_test+"patch_"+str(test_real_count)+"_"+str(i)+"_"+str(j)+".npy",patch)
							self.test_labels.append(label_parsed)
						

						test_real_count+=1
					#np.random.choice(index, samples_per_class, replace=replace)
		#==========================END IMG LOOP=============================================#
		
		if self.patches_save==True or self.patches_save=="True":
			np.save(path_save+"train_labels.npy",self.train_labels)
			np.save(path_save+"test_labels.npy",self.test_labels)
			
		print("Final mask test average",np.average(mask_test))
		cv2.imwrite("mask_train.png",mask_train)
		cv2.imwrite("mask_test.png",mask_test)
		
		deb.prints(counter,fname)
		deb.prints(patches_get["train_n"],fname)
		deb.prints(patches_get["test_n"],fname)
		deb.prints(patches_get["test_n_limited"],fname)
		deb.prints(test_real_count)
		if self.ram_store:
			if not test_only:
				#===================CLIP TRAIN DATA.================================#
				self.ram_data["train"]["n"]=patches_get["train_n"]
				self.ram_data["train"]["ims"]=self.ram_data["train"]["ims"][0:self.ram_data["train"]["n"]]
				self.ram_data["train"]["labels_int"]=self.ram_data["train"]["labels_int"][0:self.ram_data["train"]["n"]]
				count,unique=np.unique(self.ram_data["train"]["labels_int"],return_counts=True)
				print("Before squeezing count",count,unique)
				deb.prints(self.conf["squeeze_classes"])
			

			#============ CLIP TEST DATA.=============================#
			if self.conf["test"]["overlap_full"]!="True":
				self.ram_data["test"]["n"]=test_real_count
				self.ram_data["test"]["ims"]=self.ram_data["test"]["ims"][0:self.ram_data["test"]["n"]]
				self.ram_data["test"]["labels_int"]=self.ram_data["test"]["labels_int"][0:self.ram_data["test"]["n"]]
			elif test_only:
				self.ram_data["test"]["n"]=test_real_count
				self.ram_data["test"]["ims"]=self.ram_data["test"]["ims"][0:self.ram_data["test"]["n"]]
				self.ram_data["test"]["labels_int"]=self.ram_data["test"]["labels_int"][0:self.ram_data["test"]["n"]]

			unique,count=np.unique(self.ram_data["train"]["labels_int"],return_counts=True)
			print("Train patches unique",unique,count)
			unique,count=np.unique(self.ram_data["test"]["labels_int"],return_counts=True)
			print("Test patches unique",unique,count)
			
			#deb.prints()
			#===============ELIMINATE UNUSED CLASSES ==============================#

			if self.conf["squeeze_classes"]==True or self.conf["squeeze_classes"]=="True":
				print("Elminating unused classes")
				if not test_only:
					self.ram_data["train"]=self.labels_unused_classes_eliminate(self.ram_data["train"])
				if self.conf["test"]["overlap_full"]!="True" or test_only:
					self.ram_data["test"]=self.labels_unused_classes_eliminate(self.ram_data["test"],training=False)

				if no_zero==False:
					self.ram_data["test"]["labels_int"]+=1
				count,unique=np.unique(self.ram_data["train"],return_counts=True)
				#print("train count,unique",count,unique)
			#print("train count,unique",count,unique)
			unique,count=np.unique(self.ram_data["train"]["labels_int"],return_counts=True)
			print("Train patches unique",unique,count)
			unique,count=np.unique(self.ram_data["test"]["labels_int"],return_counts=True)
			print("Test patches unique",unique,count)
			

		
		return patches_get["train_n"],test_real_count
	def patches_multitemporal_get3(self,img,label,window,overlap,mask,path_train,path_test,patches_save=True, \
		label_type="one_hot",memory_mode="hdd",test_only=False, ram_store=True):

		fname=sys._getframe().f_code.co_name

		no_zero=True #No bcknd
		deb.prints(window,fname)
		deb.prints(overlap,fname)
		print("STARTED PATCH EXTRACTION")

		t_steps, h, w, channels = self.full_ims_train.shape

		indices={}
		indices['val']=np.indices((h,w))
		indices['row_flat']=indices['val'][0].flatten()
		indices['col_flat']=indices['val'][1].flatten()
		mask_flat=mask.flatten()
		label=label[self.conf['t_len']-1]
		deb.prints(label.shape)
		label_flat=label.flatten()
		deb.prints(mask_flat.shape)
		deb.prints(label_flat.shape)
		deb.prints(indices['col_flat'].shape)

		#=========== Pick useful only

		#mask_flat=mask_flat[mask_flat>0]
		mask_flat_valid=mask_flat[mask_flat>0]
		label_flat=label_flat[mask_flat>0]
		deb.prints(label_flat.shape)
		indices['row_flat']=indices['row_flat'][mask_flat>0]
		indices['col_flat']=indices['col_flat'][mask_flat>0]

		# ==== select train / test
		# Train
		self.ram_data["train"]["labels_int"]=label_flat[mask_flat_valid==1]
		indices['train_row_flat']=indices['row_flat'][mask_flat_valid==1]
		indices['train_col_flat']=indices['col_flat'][mask_flat_valid==1]
		train_mask=mask_flat[mask_flat==1]
		self.ram_data["train"]["n"]=train_mask.shape[0]

		

		# Unused classes eliminate

		data={'labels_int':self.ram_data["train"]["labels_int"]}
		self.ram_data["train"]["labels_int"]=self.labels_unused_classes_eliminate( \
			self.ram_data["train"])['labels_int']
		deb.prints(np.unique(self.ram_data["train"]["labels_int"],return_counts=True))
		
		# Test

		self.ram_data["test"]["labels_int"]=label_flat[mask_flat_valid==2]
		indices['test_row_flat']=indices['row_flat'][mask_flat_valid==2]
		indices['test_col_flat']=indices['col_flat'][mask_flat_valid==2]
		test_mask=mask_flat[mask_flat==2]
		self.ram_data["test"]["n"]=test_mask.shape[0]

		#data={'labels_int':self.ram_data["test"]["labels_int"]}
		self.ram_data["test"]["labels_int"]=self.labels_unused_classes_eliminate(self.ram_data["test"])['labels_int']
		
		deb.prints(np.unique(self.ram_data["test"]["labels_int"],return_counts=True))
		deb.prints(train_mask.shape)
		deb.prints(test_mask.shape)
		deb.prints(indices['train_row_flat'].shape)


		# ============== Here I reconstrucat the label 

		label_reconstructed=np.zeros((h,w))
		for row,col,value in zip(indices['row_flat'],
				indices['col_flat'],label_flat):
			label_reconstructed[row,col]=value
		deb.prints(np.unique(label_reconstructed,return_counts=True))

		deb.prints(label_reconstructed.dtype)
		print(label_flat.shape,label_flat.dtype)
		print(label_reconstructed.shape)
		print(np.all(label_reconstructed==label))
		print("DONE TEST")
		# ============= HERE, DO VAL / BALANCING FROM LABELS
		val_percentage=0.15

		self.ram_data['train']['ims']=np.arange(self.ram_data["train"]["labels_int"].shape[0])
		self.ram_data['train']['n']=self.ram_data['train']['labels_int'].shape[0]
		self.ram_data=val_set_get(self.ram_data,mode='stratified',validation_split=0.15)

		self.ram_data['val']['idxs']=self.ram_data['val']['ims'].copy()
		print(self.ram_data['val']['ims'][0:10])

		# Count after validation split
		self.ram_data['val']['n']=self.ram_data['val']['labels_int'].shape[0]

		# Balancing
		class_n=np.unique(self.ram_data['train']['labels_int']).shape[0]
		

		deb.prints(self.ram_data['train']['labels_int'].shape)
		self.ram_data['train']['idxs'],self.ram_data['train']['labels_int'], \
			self.ram_data['train']['labels'] = data_balance(self.ram_data, \
			self.conf["balanced"]["samples_per_class"],class_n=self.conf['class_n'])

		
		self.ram_data['train']['n']=self.ram_data['train']['labels_int'].shape[0]

		self.ram_data['train']['idxs']=self.ram_data['train']['idxs'].astype(np.int)
		self.ram_data['val']['idxs']=self.ram_data['val']['idxs'].astype(np.int)

		deb.prints(self.ram_data['train']['idxs'].dtype)
		deb.prints(self.ram_data['train']['idxs'][0:15])
		deb.prints(self.ram_data['train']['labels_int'].shape)
		deb.prints(np.unique(self.ram_data['train']['labels_int'],return_counts=True))


		self.ram_data['val']['ims']=np.zeros((self.ram_data['val']['n'],
			self.conf['t_len'],self.conf['patch']['size'],
			self.conf['patch']['size'],self.conf['band_n']))

		self.ram_data['train']['ims']=np.zeros((self.ram_data['train']['n'],
			self.conf['t_len'],self.conf['patch']['size'],
			self.conf['patch']['size'],self.conf['band_n']))

		np.save(self.conf['path']+'train_labels_int',self.ram_data['train']['labels_int'])
		np.save(self.conf['path']+'val_labels_int',self.ram_data['val']['labels_int'])
		np.save(self.conf['path']+'test_labels_int',self.ram_data['test']['labels_int'])

		print("OK")
		
		# Taking balanced indices train val

		self.ram_data['train']['row_flat']=indices['train_row_flat'][self.ram_data['train']['idxs']]
		self.ram_data['train']['col_flat']=indices['train_col_flat'][self.ram_data['train']['idxs']]

		self.ram_data['val']['row_flat']=indices['train_row_flat'][self.ram_data['val']['idxs']]
		self.ram_data['val']['col_flat']=indices['train_col_flat'][self.ram_data['val']['idxs']]
		print(self.ram_data['train']['row_flat'].shape, )
		

		np.save('locations_row.npy',indices['test_row_flat'])
		np.save('locations_col.npy',indices['test_col_flat'])
		np.save('locations_label.npy',self.ram_data["test"]["labels_int"])
		# ===== get input patches
		"""
		self.ram_self.ram_data["train"]["ims"]=np.zeros((train_mask.shape[0],
			self.conf['t_len'],self.conf['patch']['size'],
			self.conf['patch']['size'],self.conf['band_n']))


		deb.prints(self.ram_self.ram_data["train"]["ims"].shape)

		self.ram_self.ram_data["test"]["ims"]=np.zeros((test_mask.shape[0],
			self.conf['t_len'],self.conf['patch']['size'],
			self.conf['patch']['size'],self.conf['band_n']))

		deb.prints(self.ram_self.ram_data["test"]["ims"].shape)
		"""
		window_half=int(window/2)
		# Get input patches train


		#test_mode=False
		test_mode=self.test_mode
		if test_mode==True:
			print("Starting test")	
			self.bsave={}
			self.bsave['size']=1000000
			self.bsave['id']=0 # Goes from 0 to needs
			self.bsave['in_buffer']=np.zeros((self.bsave['size'],
				self.conf['t_len'],self.conf['patch']['size'],
				self.conf['patch']['size'],self.conf['band_n']))
			
			self.bsave['batch_id']=0
			
			# Get input patches test
			for row,col,count in zip(indices['test_row_flat'],
				indices['test_col_flat'],range(0,self.ram_data["test"]["n"])):

				# Reset buffer indices and store

				if count % self.bsave['size'] ==0 and count!=0:
					print(count)
					np.save(path_test+"patch"+
						str(self.bsave['batch_id'])+
						"_"+str(self.bsave['in_buffer'].shape[0])+
						".npy",self.bsave['in_buffer'])
					self.bsave['in_buffer']=np.zeros((self.bsave['size'],
						self.conf['t_len'],self.conf['patch']['size'],
						self.conf['patch']['size'],self.conf['band_n']))
					self.bsave['batch_id']+=1
					self.bsave['id']=0
				
				self.bsave['in_buffer'][self.bsave['id']]=self.full_ims_test[:,
					row-window_half:row+window_half+1,
					col-window_half:col+window_half+1,:]
				if count==0: deb.prints(self.bsave['in_buffer'][self.bsave['id']].shape)
				
				self.bsave['id']+=1		
			print("finished test loop")
			# Save last buffer
			if self.bsave['id']>0:
				self.bsave['batch_id']+=1
				self.bsave['in_buffer']=self.bsave['in_buffer'][0:self.bsave['id']]
				deb.prints(self.bsave['in_buffer'].shape)
				np.save(path_test+"patch"+
						str(self.bsave['batch_id'])+"_"+str(self.bsave['in_buffer'].shape[0])+
						".npy",self.bsave['in_buffer'])
			print("Test finished")
			self.bsave=None
			self.full_ims_test=None







		train_get=True
		if train_get==True:
			# Get train patches
			count=0
			for row,col,count in zip(self.ram_data['train']['row_flat'],
				self.ram_data['train']['col_flat'],
				range(0,self.ram_data['train']['n'])):

				# Reset buffer indices and store


				if count % 100000==0:
					print("Count",count)
				# Add a patch
				self.ram_data['train']['ims'][count]=self.full_ims_train[:,
					row-window_half:row+window_half+1,
					col-window_half:col+window_half+1,:]
				if count==0: deb.prints(self.ram_data['train']['ims'][count].shape)
			
				count+=1		
			print("Finished train loop")
			#np.save(self.conf['path']+"train_ims.npy",self.ram_data['train']['ims'])
			print("Finished saving train loop")
			#self.ram_data['train']['ims']=None
		val_get=True
		if val_get==True:
			# Get val patches
			count=0
			for row,col,count in zip(self.ram_data['val']['row_flat'],
				self.ram_data['val']['col_flat'],
				range(0,self.ram_data['val']['n'])):

				# Reset buffer indices and store


				if count % 100000==0:
					print("Count",count)
				# Add a patch
				self.ram_data['val']['ims'][count]=self.full_ims_train[:,
					row-window_half:row+window_half+1,
					col-window_half:col+window_half+1,:]
				if count==0: deb.prints(self.ram_data['val']['ims'][count].shape)
			
				count+=1		
			print("Finished val loop")
			#np.save(self.conf['path']+"val_ims.npy",self.ram_data['val']['ims'])
			print("Finished saving val")

		self.full_ims_train=False
		# test_get=False
		# if test_get==True:
		# 	print("Starting test")	
		# 	# Get input patches test
		# 	for row,col,count in zip(indices['test_row_flat'],
		# 		indices['test_col_flat'],range(0,self.ram_data["test"]["n"])):

		# 		if count%500000==0:
		# 			print("Test extract",count)
		# 		patch_test=img[:,
		# 			row-window_half:row+window_half+1,
		# 			col-window_half:col+window_half+1,:]
		# 		if count==0: deb.prints(patch_test.shape)
				
		# 		np.save(path_test+'patch'+str(count)+'.npy',patch_test)	
		# 	# Save last buffer
		# 	print("Test finished")
		# return None,None


		
		self.full_ims_test=False
		return None,None


	def in_label_ram_store(self,data,patch,label_patch,data_idx,label_type,name):
		data["ims"][data_idx]=patch
		
		if label_type=="one_hot":
			data["labels_int"][data_idx]=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
			if data["labels_int"][data_idx]==0:
				print(name)
				deb.prints("here")
		elif label_type=="semantic":
			data["labels_int"][data_idx]=label_patch[self.conf["t_len"]-1]
		return data

	def im_seq_normalize(self,im,norm_type="zscore"):
		normalize={}
		normalize["avg"]=np.zeros(self.conf["band_n"])
		normalize["std"]=np.zeros(self.conf["band_n"])
		normalize["max"]=np.zeros(self.conf["band_n"])
		normalize["min"]=np.zeros(self.conf["band_n"])
		
		#if self.debug>=1: print(data["train"]["ims"].dtype)
		for i in range(0,self.conf["band_n"]):
			if norm_type=="zscore":
				normalize["avg"][i]=np.average(im[:,:,:,i])
				normalize["std"][i]=np.std(im[:,:,:,i])
				#deb.prints(np.average(im[:,:,:,i]))
				#deb.prints(np.max(im[:,:,:,i]))
				#deb.prints(np.min(im[:,:,:,i]))
				#deb.prints(np.std(im[:,:,:,i]))
				im[:,:,:,i]=(im[:,:,:,i]-normalize["avg"][i])/normalize["std"][i]
				#deb.prints(np.average(im[:,:,:,i]))
				#deb.prints(np.max(im[:,:,:,i]))
				#deb.prints(np.min(im[:,:,:,i]))
				#deb.prints(np.std(im[:,:,:,i]))
			else:
				#im=im-np.min(im,axis=3)
				normalize["max"][i]=np.max(im[:,:,:,i])
				normalize["min"][i]=np.min(im[:,:,:,i])

				deb.prints(np.average(im[:,:,:,i]))
				deb.prints(np.max(im[:,:,:,i]))
				deb.prints(np.min(im[:,:,:,i]))
				deb.prints(np.std(im[:,:,:,i]))	
				#im[:,:,:,i]=im[:,:,:,i]-normalize["min"][i]
				im[:,:,:,i]=(im[:,:,:,i]-normalize["min"][i])/(normalize["max"][i]-normalize["min"][i])
				deb.prints(np.average(im[:,:,:,i]))
				deb.prints(np.max(im[:,:,:,i]))
				deb.prints(np.min(im[:,:,:,i]))
				deb.prints(np.std(im[:,:,:,i]))	

		deb.prints(im.shape)

		#return im.astype(np.float32)
		return im
		
	def im_seq_mask(self,im,mask):
		im_train=im.copy()
		im_test=im.copy()
		
		for band in range(0,self.conf["band_n"]):
			for t_step in range(0,self.conf["t_len"]):
				im_train[t_step,:,:,band][mask!=1]=-1
				im_test[t_step,:,:,band][mask!=2]=-1
		deb.prints(im_train.shape)
		return im_train,im_test
	def label_seq_mask(self,im,mask): 
		im=im.astype(np.uint8) 
		im_train=im.copy() 
		im_test=im.copy() 
		 
		mask_train=mask.copy() 
		mask_train[mask==2]=0 
		mask_test=mask.copy() 
		mask_test[mask==1]=0 
		mask_test[mask==2]=1 
	 
		deb.prints(im.shape) 
		deb.prints(mask_train.shape) 
	 
		deb.prints(im.dtype) 
		deb.prints(mask_train.dtype) 
		 
		im_train=cv2.bitwise_and(im,im,mask=mask_train) 
		im_test=cv2.bitwise_and(im,im,mask=mask_test) 
	 
	 
		#im_train[t_step,:,:,band][mask!=1]=-1 
		#im_test[t_step,:,:,band][mask!=2]=-1 
		deb.prints(im_train.shape) 
		return im_train,im_test
		#return im_train.astype(np.uint8),im_test.astype(np.uint8) 
	def data_normalize_per_band(self,data):
		whole_data={}
		whole_data["value"]=np.concatenate((data["train"]["ims"],data["test"]["ims"]),axis=0)
		data["normalize"]={}
		data["normalize"]["avg"]=np.zeros(self.conf["band_n"])
		data["normalize"]["std"]=np.zeros(self.conf["band_n"])
		if self.debug>=1: print(data["train"]["ims"].dtype)
		for i in range(0,self.conf["band_n"]):
			data["normalize"]["avg"][i]=np.average(whole_data["value"][:,:,:,:,i])
			data["normalize"]["std"][i]=np.std(whole_data["value"][:,:,:,:,i])
			data["train"]["ims"][:,:,:,:,i]=(data["train"]["ims"][:,:,:,:,i]-data["normalize"]["avg"][i])/data["normalize"]["std"][i]
			data["test"]["ims"][:,:,:,:,i]=(data["test"]["ims"][:,:,:,:,i]-data["normalize"]["avg"][i])/data["normalize"]["std"][i]
		return data
class DataSemantic(DataForNet):
	def __init__(self,*args,**kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing DataSemantic instance")

		deb.prints((self.conf["train"]["n_apriori"],self.conf["t_len"])+self.label_shape)

		self.ram_data["train"]["labels"]=np.zeros((self.conf["train"]["n_apriori"],self.conf["t_len"])+self.label_shape)
		self.ram_data["test"]["labels"]=np.zeros((self.conf["test"]["n_apriori"],self.conf["t_len"])+self.label_shape)
		self.ram_data["train"]["labels_int"]=np.zeros((self.conf["train"]["n_apriori"],)+self.label_shape)
		self.ram_data["test"]["labels_int"]=np.zeros((self.conf["test"]["n_apriori"],)+self.label_shape)

		self.conf["label_type"]="semantic"
		deb.prints(self.ram_data["train"]["labels"].shape)
		deb.prints(self.ram_data["test"]["labels"].shape)

	def create(self):
		os.system("rm -rf "+self.conf["path"]+"train_test")

		#os.system("rm -rf ../data/train_test")

		if self.conf["memory_mode"]=="ram":

			self.patches_load(label_type=self.conf["label_type"])

			self.ram_data["train"]["labels"]=self.ram_data["train"]["labels_int"]
			self.ram_data["test"]["labels"]=self.ram_data["test"]["labels_int"]
						

			#self.ram_data=self.data_normalize_per_band(self.ram_data)
			if self.debug>=1:
				deb.prints(self.ram_data["train"]["labels"].shape)

		else:
			print("Hdd mode not implemented yet for Im2Im data.")
			#break

	def mask_test_update(self,mask_test,yy,xx,window,label_patch,mask_patch):
		mask_test_areas=mask_patch.copy()
		mask_test_areas[mask_test_areas==1]=0 # Remove test from this patch
		mask_test_areas[mask_test_areas==2]=1 # Remove change 2 (test) to 1 values for bitwise and 
		mask_test[yy: yy + window, xx: xx + window]=mask_test_areas.astype(np.uint8)*255
		
		#deb.prints(mask_test_areas.dtype)
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A1")
		
		for t_step in range(0,self.conf["t_len"]):
			#deb.prints(label_patch[t_step].dtype)
		
			label_patch[t_step]=cv2.bitwise_and(label_patch[t_step],label_patch[t_step],mask=mask_test_areas.astype(np.uint8))
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A2")
		
		label_patch[self.conf['t_len']-1]=self.full_label_test[yy: yy + window, xx: xx + window]
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A3")
		
		return mask_test,label_patch
	def is_mask_from_train(self,mask_patch,label_patch=None):
		return np.any(mask_patch==1)
		#return np.count_nonzero(mask_patch[mask_patch==1])>64


	def is_mask_from_test(self,mask_patch,label_patch=None):
		return np.any(mask_patch==2)
		
	# def labels_unused_classes_eliminate(self,data):

	# 	data["labels"]=(data["labels"]-3).clip(min=0)
	# 	data["labels_int"]=(data["labels_int"]-3).clip(min=0)
		
	# 	return data



	def labels_unused_classes_eliminate(self,data,training=True):
		fname=sys._getframe().f_code.co_name
		##deb.prints(data["labels_int"].shape[0])
		##idxs=[i for i in range(data["labels_int"].shape[0]) if (data["labels_int"][i] == 0 or data["labels_int"][i] == 2 or data["labels_int"][i] == 3)]
		##deb.prints(len(idxs))

		# self.old_n_classes=self.n_classes
		self.classes = np.unique(data["labels_int"])
		# self.n_classes=self.classes.shape[0]
		deb.prints(self.classes,fname)		
		if training:
			self.labels2new_labels = dict((c, i) for i, c in enumerate(self.classes))
			self.new_labels2labels = dict((i, c) for i, c in enumerate(self.classes))

		new_labels = data["labels_int"].copy()
		for i in range(len(self.classes)):
			new_labels[data["labels_int"] == self.classes[i]] = self.labels2new_labels[self.classes[i]]

		# #data["old_labels_int"]=data["labels_int"].copy()
		# #data["old_labels"]=data["labels"].copy()
		
		data["labels_int"]=new_labels.copy()
		deb.prints(np.unique(data["labels_int"]))
		#data["labels"]=utils.DataOneHot.labels_onehot_get(None,data["labels_int"],data["n"],self.n_classes)
		# print(data.keys())


		return data

class DataOneHot(DataForNet):
	def __init__(self,*args,**kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing DataNetOneHot instance")
		#self.ram_data["train"]["labels"]=np.zeros((9000,)+self.label_shape)
		#self.ram_data["test"]["labels"]=np.zeros((9000,)+self.label_shape)

		self.ram_data["train"]["labels_int"]=np.zeros((self.conf["train"]["n_apriori"],))
		self.ram_data["test"]["labels_int"]=np.zeros((self.conf["test"]["n_apriori"],))
		self.conf["label_type"]="one_hot"

	def is_mask_from_train(self,mask_patch,label_patch):
		condition_1=(mask_patch[self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]==1)
		#condition_2=(label_patch[self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]>0)
		return condition_1
	def is_mask_from_test(self,mask_patch,label_patch):
		condition_1=(mask_patch[self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]==2)
		#condition_2=(label_patch[self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]]>0)
		#deb.prints(condition_1)
		#deb.prints(condition_2)
		
		#deb.prints((condition_1 and condition_2))
		return condition_1 
	def im_patches_npy_multitemporal_from_npy_from_folder_store2_onehot(self):
		#self.im_patches_npy_multitemporal_from_npy_from_folder_store2(label_type=self.conf["label_type"])
		self.patches_load(label_type=self.conf["label_type"])

		self.ram_data["train"]["labels"]=self.labels_onehot_get(self.ram_data["train"]["labels_int"], \
			self.ram_data["train"]["n"],self.conf["class_n"])
		#self.ram_data["test"]["labels"]=self.labels_onehot_get(self.ram_data["test"]["labels_int"], \
		#	self.ram_data["test"]["n"],self.conf["class_n"])
		self.ram_data["val"]["labels"]=self.labels_onehot_get(self.ram_data["val"]["labels_int"], \
			self.ram_data["val"]["n"],self.conf["class_n"])
	def create(self):
		os.system("rm -rf "+self.conf["path"]+"train_test")

#		os.system("rm -rf ../data/train_test")
		print(1)
		if self.conf["memory_mode"]=="ram":

			self.im_patches_npy_multitemporal_from_npy_from_folder_store2_onehot()
			if self.debug>=1:
				deb.prints(self.ram_data["train"]["labels_int"].shape)
			deb.prints(np.unique(self.ram_data["train"]["labels_int"],return_counts=True)[1])
			#self.ram_data=self.data_normalize_per_band(self.ram_data)
			
			data_balance=False
			if data_balance==True:
				self.ram_data["train"]["ims"],self.ram_data["train"]["labels_int"],self.ram_data["train"]["labels"]=self.data_balance(self.ram_data, \
					self.conf["balanced"]["samples_per_class"])

			#with open(self.conf["path"]+'data.pkl', 'wb') as f: pickle.dump(self.ram_data, f)

		else:
			self.im_patches_npy_multitemporal_from_npy_from_folder_store2_onehot()
			self.data_onehot_load_balance_store()

	def val_set_get(self,mode='stratified',validation_split=0.2):
		self.ram_data['train']['idx']=range(self.ram_data['train']['n'])
		clss_train_unique,clss_train_count=np.unique(self.ram_data['train']['labels_int'],return_counts=True)
		deb.prints(clss_train_count)
		self.ram_data['val']={'n':int(self.ram_data['train']['n']*validation_split)}
		
		#===== CHOOSE VAL IDX
		#mode='stratified'
		if mode=='random':
			self.ram_data['val']['idx']=np.random.choice(self.ram_data['train']['idx'],self.ram_data['val']['n'],replace=False)
			

			self.ram_data['val']['ims']=self.ram_data['train']['ims'][self.ram_data['val']['idx']]
			self.ram_data['val']['labels_int']=self.ram_data['train']['labels_int'][self.ram_data['val']['idx']]
		
		elif mode=='stratified':
			while True:
				self.ram_data['val']['idx']=np.random.choice(self.ram_data['train']['idx'],self.ram_data['val']['n'],replace=False)
				self.ram_data['val']['ims']=self.ram_data['train']['ims'][self.ram_data['val']['idx']]
				self.ram_data['val']['labels_int']=self.ram_data['train']['labels_int'][self.ram_data['val']['idx']]
		
				clss_val_unique,clss_val_count=np.unique(self.ram_data['val']['labels_int'],return_counts=True)
				
				if not np.array_equal(clss_train_unique,clss_val_unique):
					deb.prints(clss_train_unique)
					deb.prints(clss_val_unique)
					
					pass
				else:
					percentages=clss_val_count/clss_train_count
					deb.prints(percentages)
					#if np.any(percentages<0.1) or np.any(percentages>0.3):
					if np.any(percentages>0.23):
					
						pass
					else:
						break
		elif mode=='random_v2':
			while True:

				self.ram_data['val']['idx']=np.random.choice(self.ram_data['train']['idx'],self.ram_data['val']['n'],replace=False)
				

				self.ram_data['val']['ims']=self.ram_data['train']['ims'][self.ram_data['val']['idx']]
				self.ram_data['val']['labels_int']=self.ram_data['train']['labels_int'][self.ram_data['val']['idx']]
				clss_val_unique,clss_val_count=np.unique(self.ram_data['val']['labels_int'].argmax(axis=3),return_counts=True)
						
				deb.prints(clss_train_unique)
				deb.prints(clss_val_unique)

				deb.prints(clss_train_count)
				deb.prints(clss_val_count)

				clss_train_count_in_val=clss_train_count[np.isin(clss_train_unique,clss_val_unique)]
				percentages=clss_val_count/clss_train_count_in_val
				deb.prints(percentages)
				#if np.any(percentages<0.1) or np.any(percentages>0.3):
				if np.any(percentages>0.26):
					pass
				else:
					break				

		deb.prints(self.ram_data['val']['idx'].shape)

		
		deb.prints(self.ram_data['val']['ims'].shape)
		#deb.prints(data.patches['val']['labels_int'].shape)
		
		self.ram_data['train']['ims']=np.delete(self.ram_data['train']['ims'],self.ram_data['val']['idx'],axis=0)
		self.ram_data['train']['labels_int']=np.delete(self.ram_data['train']['labels_int'],self.ram_data['val']['idx'],axis=0)
		self.ram_data['train']['n']=self.ram_data['train']['ims'].shape[0]
		self.ram_data['val']['n']=self.ram_data['val']['ims'].shape[0]
		
		#deb.prints(data.patches['train']['ims'].shape)
		#deb.prints(data.patches['train']['labels_int'].shape)

	def mask_test_update(self,mask_test,yy,xx,window,label_patch,mask_patch):
		
		#mask_test_areas=mask_patch.copy()
		#mask_test_areas[mask_test_areas==1]=0 # Remove test from this patch
		#mask_test_areas[mask_test_areas==2]=1 # Remove change 2 (test) to 1 values for bitwise and 
		mask_test[self.conf["patch"]["center_pixel"], self.conf["patch"]["center_pixel"]]=255
		"""
		#mask_test[yy: yy + window, xx: xx + window]=mask_test_areas.astype(np.uint8)*255
		
		#deb.prints(mask_test_areas.dtype)
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A1")
		
		for t_step in range(0,self.conf["t_len"]):
			#deb.prints(label_patch[t_step].dtype)
		
			label_patch[t_step]=cv2.bitwise_and(label_patch[t_step],label_patch[t_step],mask=mask_test_areas.astype(np.uint8))
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A2")
		
		label_patch[self.conf['t_len']-1]=self.full_label_test[yy: yy + window, xx: xx + window]
		center_label=int(label_patch[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
		if center_label==0:
			print("A3")
		"""
		#print("here",int(yy + window/2),int(xx + window/2))
		return mask_test,label_patch

	def data_onehot_load_balance_store(self):
		print("heeere")
		self.conf["train"]["n"]=np.load(self.conf["path"]+"train_n.npy")
		self.conf["test"]["n"]=np.load(self.conf["path"]+"test_n.npy")
		deb.prints(self.conf["train"]["n"])
		deb.prints(self.conf["test"]["n"])


		data=self.im_patches_npy_multitemporal_from_npy_from_folder_load2()
		
		deb.prints(data["train"]["ims"].shape)
		deb.prints(data["test"]["ims"].shape)
		
		classes,counts=np.unique(data["train"]["labels_int"],return_counts=True)
		print(classes,counts)

		data=self.data_normalize_per_band(data)
		if self.conf["pc_mode"]=="remote":
			samples_per_class=500
		else:
			samples_per_class=5000

		data["train"]["ims"],data["train"]["labels_int"],data["train"]["labels"]=self.data_balance(data,self.conf["balanced"]["samples_per_class"])
		data["train"]["n"]=data["train"]["ims"].shape[0]

		deb.prints(data["train"]["ims"].shape)
		deb.prints(data["test"]["ims"].shape)

		filename = self.conf["path"]+'data.pkl'
		#os.makedirs(os.path.dirname(filename), exist_ok=True)
		print(list(iter(data)))
		if self.conf["utils_flag_store"]:
			# Store data train, test ims and labels_one_hot
			os.system("rm -rf "+self.conf["path"]+"balanced")
#			os.system("rm -rf ../data/balanced")
			self.data_save_to_npy(self.conf["train"],data["train"])
			self.data_save_to_npy(self.conf["test"],data["test"])	

	def im_patches_npy_multitemporal_from_npy_from_folder_load2(self,load=True,debug=1):
		fname=sys._getframe().f_code.co_name
		
		data={}
		data["train"]={}
		data["test"]={}

		if load:
			data["train"]=self.im_patches_labelsonehot_load2(self.conf["train"],data["train"],data,debug=0)
			data["test"]=self.im_patches_labelsonehot_load2(self.conf["test"],data["test"],data,debug=0)
		else:
			train_ims=[]
		deb.prints(list(iter(data["train"])))

		#np.save(self.conf["path"]+"data.npy",data) # Save indexes and data for further use with the train/ test set/labels
		return data	

	# From data_get()
	def labels_unused_classes_eliminate(self,data,training=True):
		fname=sys._getframe().f_code.co_name
		##deb.prints(data["labels_int"].shape[0])
		##idxs=[i for i in range(data["labels_int"].shape[0]) if (data["labels_int"][i] == 0 or data["labels_int"][i] == 2 or data["labels_int"][i] == 3)]
		##deb.prints(len(idxs))

		# self.old_n_classes=self.n_classes
		self.classes = np.unique(data["labels_int"])
		# self.n_classes=self.classes.shape[0]
		deb.prints(self.classes,fname)		
		
		if training:
			self.labels2new_labels = dict((c, i) for i, c in enumerate(self.classes))
			self.new_labels2labels = dict((i, c) for i, c in enumerate(self.classes))

		new_labels = data["labels_int"].copy()
		for i in range(len(self.classes)):
			new_labels[data["labels_int"] == self.classes[i]] = self.labels2new_labels[self.classes[i]]

		# #data["old_labels_int"]=data["labels_int"].copy()
		# #data["old_labels"]=data["labels"].copy()
		
		data["labels_int"]=new_labels.copy()
		deb.prints(np.unique(data["labels_int"]))
		#data["labels"]=utils.DataOneHot.labels_onehot_get(None,data["labels_int"],data["n"],self.n_classes)
		# print(data.keys())


		return data
	def unused_classes_elimination_configure(self,label):
		fname=sys._getframe().f_code.co_name
		##deb.prints(data["labels_int"].shape[0])
		##idxs=[i for i in range(data["labels_int"].shape[0]) if (data["labels_int"][i] == 0 or data["labels_int"][i] == 2 or data["labels_int"][i] == 3)]
		##deb.prints(len(idxs))

		# self.old_n_classes=self.n_classes
		self.classes = np.unique(label)[1:] #No bcknd
		# self.n_classes=self.classes.shape[0]
		deb.prints(self.classes,fname)		
		self.labels2new_labels = dict((c, i) for i, c in enumerate(self.classes))
		self.new_labels2labels = dict((i, c) for i, c in enumerate(self.classes))

	def labels_unused_classes_eliminate_prior(self,label,unused_map=[[0,0],
					[1,1],
					[2,2],
					[3,3],
					[4,4],
					[5,-1],
					[6,5],
					[7,6],
					[8,7],
					[9,8],
					[10,9],
					[11,10]]):
		fname=sys._getframe().f_code.co_name

		new_label=self.labels2new_labels[label]

		##new_labels = labels_int.copy()
		##for i in range(len(self.classes)):
		##	new_labels[labels_int == self.classes[i]] = self.labels2new_labels[self.classes[i]]

		# #data["old_labels_int"]=data["labels_int"].copy()
		# #data["old_labels"]=data["labels"].copy()
		
		##data["labels_int"]=new_labels.copy()
		##deb.prints(np.unique(data["labels_int"]))
		#data["labels"]=utils.DataOneHot.labels_onehot_get(None,data["labels_int"],data["n"],self.n_classes)
		# print(data.keys())


		return new_label



	def data_balance(self, data, samples_per_class):
		fname=sys._getframe().f_code.co_name

		balance={}
		balance["unique"]={}
	#	classes = range(0,self.conf["class_n"])
		classes,counts=np.unique(data["train"]["labels_int"],return_counts=True)
		print(classes,counts)
		deb.prints(self.conf["squeeze_classes"])
		if self.conf["squeeze_classes"]==False:
			if classes[0]==0: classes=classes[1::]
		num_total_samples=len(classes)*samples_per_class
		balance["out_labels"]=np.zeros(num_total_samples)
		deb.prints((num_total_samples,) + data["train"]["ims"].shape[1::],fname)
		deb.prints(num_total_samples)
		balance["out_data"]=np.zeros((num_total_samples,) + data["train"]["ims"].shape[1::])
		
		#balance["unique"]=dict(zip(unique, counts))
		#print(balance["unique"])
		k=0
		for clss in classes:
			if self.conf["squeeze_classes"]==False:
				if clss==0: 
					continue
			deb.prints(clss,fname)
			balance["data"]=data["train"]["ims"][data["train"]["labels_int"]==clss]
			balance["labels_int"]=data["train"]["labels_int"][data["train"]["labels_int"]==clss]
			balance["num_samples"]=balance["data"].shape[0]
			if self.debug>=1: deb.prints(balance["data"].shape,fname)
			if self.debug>=2: 
				deb.prints(balance["labels_int"].shape,fname)
				deb.prints(np.unique(balance["labels_int"].shape),fname)
			if balance["num_samples"] > samples_per_class:
				replace=False
			else: 
				replace=True

			index = range(balance["labels_int"].shape[0])
			index = np.random.choice(index, samples_per_class, replace=replace)
			balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["labels_int"][index]
			balance["out_data"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["data"][index]

			k+=1
		idx = np.random.permutation(balance["out_labels"].shape[0])
		balance["out_data"] = balance["out_data"][idx]
		balance["out_labels"] = balance["out_labels"][idx]

		balance["labels"]=self.labels_onehot_get(balance["out_labels"],num_total_samples,self.conf["class_n"])
		#balance["labels"]=np.zeros((num_total_samples,self.conf["class_n"]))
		#balance["labels"][np.arange(num_total_samples),balance["out_labels"].astype(np.int)]=1
		if self.debug>=1: deb.prints(np.unique(balance["out_labels"],return_counts=True),fname)
		return balance["out_data"],balance["out_labels"],balance["labels"]

	def labels_onehot_get(self,labels,n_samples,class_n):
		out=np.zeros((n_samples,class_n))
		deb.prints(out.shape)
		deb.prints(labels.shape)
		out[np.arange(n_samples),labels.astype(np.int)]=1
		return out


	def data_save_to_npy(self,conf_set,data):
		pathlib.Path(conf_set["balanced_path"]+"label/").mkdir(parents=True, exist_ok=True) 
		pathlib.Path(conf_set["balanced_path"]+"ims/").mkdir(parents=True, exist_ok=True) 
		
		for i in range(0,data["ims"].shape[0]):
			np.save(conf_set["balanced_path"]+"ims/"+"patch_"+str(i)+".npy",data["ims"][i])
		np.save(conf_set["balanced_path"]+"label/labels.npy",data["labels"])			

	# Use after patches_npy_multitemporal_from_npy_store2(). Probably use within patches_npy_multitemporal_from_npy_from_folder_load2()
	def im_patches_labelsonehot_load2(self,conf_set,data,data_whole,debug=0): #data["n"], self.conf["patch"]["ims_path"], self.conf["patch"]["labels_path"]
		print("[@im_patches_labelsonehot_load2]")
		fname=sys._getframe().f_code.co_name

		data["ims"]=np.zeros((conf_set["n"],self.conf["t_len"],self.conf["patch"]["size"],self.conf["patch"]["size"],self.conf["band_n"]))
		data["labels_int"]=np.zeros((conf_set["n"])).astype(np.int)
			
		count=0
		if self.debug>=1: deb.prints(conf_set["ims_path"],fname)
		for i in range(1,conf_set["n"]):
			if self.debug>=3: print("i",i)
			im_name=glob.glob(conf_set["ims_path"]+'patch_'+str(i)+'_*')[0]
			data["ims"][count,:,:,:,:]=np.load(im_name)
			
			label_name=glob.glob(conf_set["labels_path"]+'patch_'+str(i)+'_*')[0]
			data["labels_int"][count]=int(np.load(label_name)[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
			if self.debug>=2: print("train_labels[count]",data["labels_int"][count])
			
			count=count+1
			if i % 1000==0:
				print("file ID",i)
		data["labels"]=np.zeros((conf_set["n"],self.conf["class_n"]))
		data["labels"][np.arange(conf_set["n"]),data["labels_int"]]=1
		#del data["labels_int"]
		return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--debug', type=int, default=1, help='Debug')
	parser.add_argument('-po','--patch_overlap', dest='patch_overlap', type=int, default=0, help='Debug')
	parser.add_argument('--im_size', dest='im_size', type=int, default=(948,1068), help='Debug')
	parser.add_argument('--band_n', dest='band_n', type=int, default=7, help='Debug')
	parser.add_argument('--t_len', dest='t_len', type=int, default=6, help='Debug')
	parser.add_argument('--path', dest='path', default="../data/", help='Data path')
	parser.add_argument('--class_n', dest='class_n', type=int, default=9, help='Class number')
	parser.add_argument('--pc_mode', dest='pc_mode', default="local", help="Class number. 'local' or 'remote'")
	parser.add_argument('-tnl','--test_n_limit', dest='test_n_limit',type=int, default=1000, help="Class number. 'local' or 'remote'")
	parser.add_argument('-mm','--memory_mode', dest='memory_mode',default="ram", help="Class number. 'local' or 'remote'")
	parser.add_argument('-bs','--balance_samples_per_class', dest='balance_samples_per_class',type=int,default=None, help="Class number. 'local' or 'remote'")
	parser.add_argument('-ts','--test_get_stride', dest='test_get_stride',type=int,default=8, help="Class number. 'local' or 'remote'")
	parser.add_argument('-nap','--n_apriori', dest='n_apriori',type=int,default=5000000, help="Class number. 'local' or 'remote'")
	parser.add_argument('-sc','--squeeze_classes', dest='squeeze_classes',default=True, help="Class number. 'local' or 'remote'")

	args = parser.parse_args()
	patch_length=5
	
	#data=DataOneHot(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
	
	data=DataSemantic(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
		band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
		test_n_limit=args.test_n_limit, memory_mode=args.memory_mode, flag_store=True, \
		balance_samples_per_class=args.balance_samples_per_class, test_get_stride=args.test_get_stride, \
		n_apriori=args.n_apriori,patch_length=patch_length, squeeze_classes=args.squeeze_classes)
	data.create()
	

	##data=DataForNet()
	##data.im_npy_get()
	#conf=data_creator.conf
	#pass
#else:
	#data_onehot=DataOneHot()
	##data_onehot=DataSemantic()
	
	#data.onehot_create()
	##conf=data_onehot.conf
	
