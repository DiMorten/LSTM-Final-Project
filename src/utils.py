
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
#from skimage.transform import resize
#from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import cv2
import pathlib
#from sklearn.feature_extraction.image import extract_patches_2d
#from skimage.util import view_as_windows
import sys
import pickle
# Local
import deb
import argparse



class DataForNet(object):
	def __init__(self,debug=1,patch_overlap=0,im_size=(948,1068),band_n=6,t_len=6,path="../data/",class_n=9,pc_mode="local", \
		patch_length=5,test_n_limit=1000,data_memory_mode="hdd"):
		self.data_memory_mode=data_memory_mode #"ram" or "hdd"
		self.debug=debug
		self.test_n_limit=test_n_limit
		self.conf={"band_n": band_n, "t_len":t_len, "path": path, "class_n":class_n}

		self.conf["pc_mode"]=pc_mode

		self.conf["out_path"]=self.conf["path"]+"results/"
		self.conf["in_npy_path"]=self.conf["path"]+"in_npy/"
		self.conf["in_rgb_path"]=self.conf["path"]+"in_rgb/"
		self.conf["in_labels_path"]=self.conf["path"]+"labels/"
		self.conf["patch"]={}
		self.conf["patch"]={"size":patch_length, "stride":5, "out_npy_path":self.conf["path"]+"patches_npy/"}
		self.conf["patch"]["ims_path"]=self.conf["patch"]["out_npy_path"]+"patches_all/"
		self.conf["patch"]["labels_path"]=self.conf["patch"]["out_npy_path"]+"labels_all/"
		self.conf['patch']['center_pixel']=int(np.around(self.conf["patch"]["size"]/2))
		self.conf["train"]={}
		self.conf["train"]["mask"]={}
		self.conf["train"]["mask"]["dir"]=self.conf["path"]+"TrainTestMask.tif"
		self.conf["train"]["ims_path"]=self.conf["path"]+"train_test/train/ims/"
		self.conf["train"]["labels_path"]=self.conf["path"]+"train_test/train/labels/"
		self.conf["test"]={}
		self.conf["test"]["ims_path"]=self.conf["path"]+"train_test/test/ims/"
		self.conf["test"]["labels_path"]=self.conf["path"]+"train_test/test/labels/"
		self.conf["im_size"]=im_size
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
			self.conf["balanced"]["samples_per_class"]=500

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

		deb.prints(self.conf["patch"]["overlap"])
		deb.prints(self.conf["extract"]["test_skip"])
		deb.prints(self.conf["balanced"]["samples_per_class"])
		

		pathlib.Path(self.conf["train"]["balanced_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["test"]["balanced_path"]).mkdir(parents=True, exist_ok=True) 

		self.conf["utils_main_mode"]=7
		self.conf["utils_flag_store"]=True

		print(self.conf)

class DataOneHot(DataForNet):
	def __init__(self,*args,**kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing DataNetOneHot instance")


	def onehot_create(self):
		os.system("rm -rf ../data/train_test")

		self.im_patches_npy_multitemporal_from_npy_from_folder_store2()
		self.data_onehot_load_balance_store()
	def im_patches_npy_multitemporal_from_npy_from_folder_store2(self):
		im_names=[]
		for i in range(1,10):
			im_name=glob.glob(self.conf["in_npy_path"]+'im'+str(i)+'*')[0]
			im_name=im_name[-14:-4]
			im_names.append(im_name)
			print(im_name)
		print(im_names)
		self.im_patches_npy_multitemporal_from_npy_store2(im_names)
	def im_patches_npy_multitemporal_from_npy_store2(self,names,train_mask_save=True):
		fname=sys._getframe().f_code.co_name
		print('[@im_patches_npy_multitemporal_from_npy_store2]')
		patch_shape=(self.conf["patch"]["size"],self.conf["patch"]["size"],self.conf["band_n"])
		label_shape=(self.conf["patch"]["size"],self.conf["patch"]["size"])
		pathlib.Path(self.conf["patch"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["patch"]["labels_path"]).mkdir(parents=True, exist_ok=True) 
		patches_all=np.zeros((58,65,self.conf["t_len"])+patch_shape)
		label_patches_all=np.zeros((58,65,self.conf["t_len"])+label_shape)
		
		patch={}
		deb.prints((self.conf["t_len"],)+patch_shape)

		#patch["values"]=np.zeros((self.conf["t_len"],)+patch_shape)
		patch["full_ims"]=np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"])
		patch["full_label_ims"]=np.zeros((self.conf["t_len"],)+self.conf["im_3d_size"][0:2])
		#for t_step in range(0,self.conf["t_len"]):
		for t_step in range(0,self.conf["t_len"]):	
			deb.prints(self.conf["in_npy_path"]+names[t_step+3]+".npy")
			patch["full_ims"][t_step] = np.load(self.conf["in_npy_path"]+names[t_step+3]+".npy")
			patch["full_label_ims"][t_step] = cv2.imread(self.conf["path"]+"labels/"+names[t_step+3][2]+".tif",0)

		deb.prints(patch["full_ims"].shape,fname)
		deb.prints(patch["full_label_ims"].shape,fname)

		# Load train mask
		#self.conf["patch"]["overlap"]=26

		pathlib.Path(self.conf["train"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["train"]["labels_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["test"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(self.conf["test"]["labels_path"]).mkdir(parents=True, exist_ok=True) 

		patch["train_mask"]=cv2.imread(self.conf["train"]["mask"]["dir"],0)

		
		self.conf["train"]["n"],self.conf["test"]["n"]=self.patches_multitemporal_get(patch["full_ims"],patch["full_label_ims"],self.conf["patch"]["size"],self.conf["patch"]["overlap"], \
			mask=patch["train_mask"],path_train=self.conf["train"],path_test=self.conf["test"],patches_save=self.conf["utils_flag_store"])
		deb.prints(self.conf["test"]["n"])
		np.save(self.conf["path"]+"train_n.npy",self.conf["train"]["n"])
		np.save(self.conf["path"]+"test_n.npy",self.conf["test"]["n"])

	def patches_multitemporal_get(self,img,label,window,overlap,mask,path_train,path_test,patches_save=False):
		fname=sys._getframe().f_code.co_name
		deb.prints(window,fname)
		deb.prints(overlap,fname)
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
		counter=0
		patches_get["train_n"]=0
		patches_get["test_n"]=0
		patches_get["test_n_limited"]=0
		test_counter=0
		test_real_count=0
		for i in range(len(gridx)):
			for j in range(len(gridy)):
				counter=counter+1
				xx = gridx[i]
				yy = gridy[j]
				#patch_clouds=Bclouds[yy: yy + window, xx: xx + window]
				patch = img[:,yy: yy + window, xx: xx + window,:]
				label_patch = label[:,yy: yy + window, xx: xx + window]
				mask_patch = mask[yy: yy + window, xx: xx + window]
				if np.any(label_patch==0):
					continue
				elif np.all(mask_patch==1): # Train sample
					patches_get["train_n"]+=1
					mask_train[yy: yy + window, xx: xx + window]=255
					if self.data_memory_mode=="hdd":
						if patches_save==True:
							np.save(path_train["ims_path"]+"patch_"+str(patches_get["train_n"])+"_"+str(i)+"_"+str(j)+".npy",patch)
							np.save(path_train["labels_path"]+"patch_"+str(patches_get["train_n"])+"_"+str(i)+"_"+str(j)+".npy",label_patch)
					#elif self.data_memory_mode=="ram":
				#		self.data["ims"]
				elif np.all(mask_patch==2): # Test sample
					test_counter+=1
					
					#if np.random.rand(1)[0]>=0.7:
					
					patches_get["test_n"]+=1
					if patches_get["test_n"]<=self.test_n_limit:
						patches_get["test_n_limited"]+=1
						if patches_save==True:
							if test_counter>=self.conf["extract"]["test_skip"]:
								mask_test[yy: yy + window, xx: xx + window]=255
								test_counter=0
								test_real_count+=1
								if self.data_memory_mode=="hdd":
									np.save(path_test["ims_path"]+"patch_"+str(test_real_count)+"_"+str(i)+"_"+str(j)+".npy",patch)
									np.save(path_test["labels_path"]+"patch_"+str(test_real_count)+"_"+str(i)+"_"+str(j)+".npy",label_patch)
					#np.random.choice(index, samples_per_class, replace=replace)
		cv2.imwrite("mask_train.png",mask_train)
		cv2.imwrite("mask_test.png",mask_test)
		
		deb.prints(counter,fname)
		deb.prints(patches_get["train_n"],fname)
		deb.prints(patches_get["test_n"],fname)
		deb.prints(patches_get["test_n_limited"],fname)
		deb.prints(test_real_count)
		
		return patches_get["train_n"],test_real_count
	def data_onehot_load_balance_store(self):
		print("heeere")
		self.conf["train"]["n"]=np.load(self.conf["path"]+"train_n.npy")
		self.conf["test"]["n"]=np.load(self.conf["path"]+"test_n.npy")
		deb.prints(self.conf["train"]["n"])
		deb.prints(self.conf["test"]["n"])


		data=self.im_patches_npy_multitemporal_from_npy_from_folder_load2()
		
		deb.prints(data["train"]["ims"].shape)
		deb.prints(data["test"]["ims"].shape)
		
		classes,counts=np.unique(data["train"]["labels"],return_counts=True)
		print(classes,counts)

		data=self.data_normalize_per_band(data)
		if self.conf["pc_mode"]=="remote":
			samples_per_class=500
		else:
			samples_per_class=5000

		data["train"]["ims"],data["train"]["labels"],data["train"]["labels_onehot"]=self.data_balance(data,self.conf["balanced"]["samples_per_class"])
		data["train"]["n"]=data["train"]["ims"].shape[0]

		deb.prints(data["train"]["ims"].shape)
		deb.prints(data["test"]["ims"].shape)

		filename = self.conf["path"]+'data.pkl'
		#os.makedirs(os.path.dirname(filename), exist_ok=True)
		print(list(iter(data)))
		if self.conf["utils_flag_store"]:
			# Store data train, test ims and labels_one_hot
			os.system("rm -rf ../data/balanced")
			self.data_save_to_npy(self.conf["train"],data["train"])
			self.data_save_to_npy(self.conf["test"],data["test"])	
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
	def data_balance(self, data, samples_per_class):
		fname=sys._getframe().f_code.co_name

		balance={}
		balance["unique"]={}
	#	classes = range(0,self.conf["class_n"])
		classes,counts=np.unique(data["train"]["labels"],return_counts=True)
		print(classes,counts)
		
		if classes[0]==0: classes=classes[1::]
		num_total_samples=len(classes)*samples_per_class
		balance["out_labels"]=np.zeros(num_total_samples)
		deb.prints((num_total_samples,) + data["train"]["ims"].shape[1::],fname)
		balance["out_data"]=np.zeros((num_total_samples,) + data["train"]["ims"].shape[1::])
		
		#balance["unique"]=dict(zip(unique, counts))
		#print(balance["unique"])
		k=0
		for clss in classes:
			if clss==0: 
				continue
			deb.prints(clss,fname)
			balance["data"]=data["train"]["ims"][data["train"]["labels"]==clss]
			balance["labels"]=data["train"]["labels"][data["train"]["labels"]==clss]
			balance["num_samples"]=balance["data"].shape[0]
			if self.debug>=1: deb.prints(balance["data"].shape,fname)
			if self.debug>=2: 
				deb.prints(balance["labels"].shape,fname)
				deb.prints(np.unique(balance["labels"].shape),fname)
			if balance["num_samples"] > samples_per_class:
				replace=False
			else: 
				replace=True

			index = range(balance["labels"].shape[0])
			index = np.random.choice(index, samples_per_class, replace=replace)
			balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["labels"][index]
			balance["out_data"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["data"][index]

			k+=1
		idx = np.random.permutation(balance["out_labels"].shape[0])
		balance["out_data"] = balance["out_data"][idx]
		balance["out_labels"] = balance["out_labels"][idx]
		balance["labels_onehot"]=np.zeros((num_total_samples,self.conf["class_n"]))
		balance["labels_onehot"][np.arange(num_total_samples),balance["out_labels"].astype(np.int)]=1
		if self.debug>=1: deb.prints(np.unique(balance["out_labels"],return_counts=True),fname)
		return balance["out_data"],balance["out_labels"],balance["labels_onehot"]
	def data_save_to_npy(self,conf_set,data):
		pathlib.Path(conf_set["balanced_path"]+"label/").mkdir(parents=True, exist_ok=True) 
		pathlib.Path(conf_set["balanced_path"]+"ims/").mkdir(parents=True, exist_ok=True) 
		
		for i in range(0,data["ims"].shape[0]):
			np.save(conf_set["balanced_path"]+"ims/"+"patch_"+str(i)+".npy",data["ims"][i])
		np.save(conf_set["balanced_path"]+"label/labels.npy",data["labels_onehot"])			

	# Use after patches_npy_multitemporal_from_npy_store2(). Probably use within patches_npy_multitemporal_from_npy_from_folder_load2()
	def im_patches_labelsonehot_load2(self,conf_set,data,data_whole,debug=0): #data["n"], self.conf["patch"]["ims_path"], self.conf["patch"]["labels_path"]
		print("[@im_patches_labelsonehot_load2]")
		fname=sys._getframe().f_code.co_name

		data["ims"]=np.zeros((conf_set["n"],self.conf["t_len"],self.conf["patch"]["size"],self.conf["patch"]["size"],self.conf["band_n"]))
		data["labels"]=np.zeros((conf_set["n"])).astype(np.int)
			
		count=0
		if self.debug>=1: deb.prints(conf_set["ims_path"],fname)
		for i in range(1,conf_set["n"]):
			if self.debug>=3: print("i",i)
			im_name=glob.glob(conf_set["ims_path"]+'patch_'+str(i)+'_*')[0]
			data["ims"][count,:,:,:,:]=np.load(im_name)
			
			label_name=glob.glob(conf_set["labels_path"]+'patch_'+str(i)+'_*')[0]
			data["labels"][count]=int(np.load(label_name)[self.conf["t_len"]-1,self.conf["patch"]["center_pixel"],self.conf["patch"]["center_pixel"]])
			if self.debug>=2: print("train_labels[count]",data["labels"][count])
			
			count=count+1
			if i % 1000==0:
				print("file ID",i)
		data["labels_onehot"]=np.zeros((conf_set["n"],self.conf["class_n"]))
		data["labels_onehot"][np.arange(conf_set["n"]),data["labels"]]=1
		#del data["labels"]
		return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--debug', type=int, default=1, help='Debug')
	parser.add_argument('-po','--patch_overlap', dest='patch_overlap', type=int, default=0, help='Debug')
	parser.add_argument('--im_size', dest='im_size', type=int, default=(948,1068), help='Debug')
	parser.add_argument('--band_n', dest='band_n', type=int, default=6, help='Debug')
	parser.add_argument('--t_len', dest='t_len', type=int, default=6, help='Debug')
	parser.add_argument('--path', dest='path', default="../data/", help='Data path')
	parser.add_argument('--class_n', dest='class_n', type=int, default=9, help='Class number')
	parser.add_argument('--pc_mode', dest='pc_mode', default="local", help="Class number. 'local' or 'remote'")
	parser.add_argument('-tnl','--test_n_limit', dest='test_n_limit',type=int, default=500000, help="Class number. 'local' or 'remote'")

	args = parser.parse_args()

	data_creator=DataOneHot(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
		band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
		test_n_limit=args.test_n_limit)
	data_creator.onehot_create()
	#conf=data_creator.conf
	#pass
else:
	data_creator=DataOneHot()
	conf=data_creator.conf
	