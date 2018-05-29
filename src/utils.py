 
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
import cv2
import pathlib
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import view_as_windows


def im_store_patches_npy(path,name,band_n,out_path,in_rgb=False):
	im = load_landsat(path,band_n)
	pathlib.Path(path+"patches/"+name).mkdir(parents=True, exist_ok=True) 


def im_patches_npy_multitemporal_from_npy_store(conf,names):
	patch_shape=(conf["patch"]["size"],conf["patch"]["size"],conf["band_n"])
	label_shape=(conf["patch"]["size"],conf["patch"]["size"])
	pathlib.Path(conf["patch"]["ims_path"]).mkdir(parents=True, exist_ok=True) 
	pathlib.Path(conf["patch"]["labels_path"]).mkdir(parents=True, exist_ok=True) 
	patches_all=np.zeros((58,65,conf["t_len"])+patch_shape)
	label_patches_all=np.zeros((58,65,conf["t_len"])+label_shape)
	
	for i in range(1,10):
		im = np.load(conf["in_npy_path"]+names[i-1]+".npy")
		print(names[i-1])
		labels = cv2.imread(conf["path"]+"labels/"+names[i-1][2]+".tif",0)
		print("labels shape",labels.shape)
		print("im shape",im.shape)
		
		patches=np.squeeze(view_as_windows(im,patch_shape,step=conf["patch"]["stride"]))
		print("label_shape",label_shape)
		label_patches=view_as_windows(labels,label_shape,step=conf["patch"]["stride"])
		label_patches=np.squeeze(label_patches)
		patches_all[:,:,i-1,:,:,:]=patches.copy()
		label_patches_all[:,:,i-1,:,:]=label_patches.copy()

	print(conf["patch"]["ims_path"])
	print("patches",patches_all.shape)
	print("label_patches.s",label_patches_all.shape)
	count=0
	for i in range(patches.shape[0]):
		for k in range(patches.shape[1]):
			#pass
			np.save(conf["patch"]["ims_path"]+"patch_"+str(count)+"_"+str(i)+"_"+str(k)+".npy",patches_all[i,k,:,:,:,:])
			np.save(conf["patch"]["labels_path"]+"patch_"+str(count)+"_"+str(i)+"_"+str(k)+".npy",label_patches_all[i,k,:,:,:])
			count=count+1

def im_store_patches_npy_from_npy(conf,name):
	out_path={"patches":conf["patch"]["out_npy_path"]+"im/"+name+"/","labels":conf["patch"]["out_npy_path"]+"labels/"+name+"/"}
	patch_shape=(conf["patch"]["size"],conf["patch"]["size"],conf["band_n"])
	label_shape=(conf["patch"]["size"],conf["patch"]["size"])

	im = np.load(conf["in_npy_path"]+name+".npy")
	print(name)
	labels = cv2.imread(conf["path"]+"labels/"+name[2]+".tif",0)
	print("labels shape",labels.shape)
	print("im shape",im.shape)


	
	patches=np.squeeze(view_as_windows(im,patch_shape,step=conf["patch"]["stride"]))
	print("label_shape",label_shape)
	label_patches=view_as_windows(labels,label_shape,step=conf["patch"]["stride"])
	label_patches=np.squeeze(label_patches)
	
	print(out_path["patches"])
	pathlib.Path(out_path["patches"]).mkdir(parents=True, exist_ok=True) 
	pathlib.Path(out_path["labels"]).mkdir(parents=True, exist_ok=True) 
	print("patches",patches.shape)
	print("label_patches.s",label_patches.shape)
	
	for i in range(patches.shape[0]):
		for k in range(patches.shape[1]):
			#pass
			np.save(out_path["patches"]+"patch_"+str(i)+"_"+str(k)+".npy",patches[i,k,:,:,:])
			np.save(out_path["labels"]+"patch_"+str(i)+"_"+str(k)+".npy",label_patches[i,k,:,:])

def im_store_patches_npy_from_npy_from_folder(conf):
	#ims["full"]=[]
	for i in range(1,10):
		im_name=glob.glob(conf["in_npy_path"]+'im'+str(i)+'*')[0]
		im_name=im_name[-14:-4]
		print(im_name)
		im_store_patches_npy_from_npy(conf,im_name)
		
		#ims["full"].append(np.load(im_name))
	#return ims

def im_patches_npy_multitemporal_from_npy_from_folder_store(conf):
	#ims["full"]=[]
	im_names=[]
	for i in range(0,10):
		im_name=glob.glob(conf["in_npy_path"]+'im'+str(i)+'*')[0]
		im_name=im_name[-14:-4]
		im_names.append(im_name)
		print(im_name)
	print(im_names)
	im_patches_npy_multitemporal_from_npy_store(conf,im_names)
		
		#ims["full"].append(np.load(im_name))
	#return ims


def im_patches_npy_multitemporal_from_npy_from_folder_load(conf,train_test_split=True,train_percentage=0.01,load=True,debug=1):
	#ims["full"]=[]
	data={"train_test_split":train_test_split,"im_n":3769}
	data["index"]=np.arange(0,data["im_n"])
	if debug>=1: print("data[index]",data["index"].shape)
	data["index_shuffle"]=data["index"].copy()
	np.random.shuffle(data["index_shuffle"])
	if debug>=2: print("data[index]shufflle",data["index_shuffle"].shape)

	data["train"]={"n":int(np.around(data["im_n"]*train_percentage))}

	data["train"]["index"]=data["index_shuffle"][0:data["train"]["n"]]
	np.save("data.npy",data) # Save indexes for further use with the test set
	if debug>=1: print(data["train"]["index"])
	if debug>=1: print("Patches location",conf["patch"]["ims_path"])
	if load:
		data["train"]["ims"]=np.zeros((data["train"]["n"],conf["t_len"],conf["patch"]["size"],conf["patch"]["size"],conf["band_n"]))
		data["train"]["labels"]=np.zeros((data["train"]["n"])).astype(np.int)
		count=0
		for i in data["train"]["index"]:
			#print("i",i)
			im_name=glob.glob(conf["patch"]["ims_path"]+'/patch_'+str(i)+'_*')[0]
			data["train"]["ims"][count,:,:,:,:]=np.load(im_name)
			
			label_name=glob.glob(conf["patch"]["labels_path"]+'/patch_'+str(i)+'_*')[0]
			data["train"]["labels"][count]=int(np.load(label_name)[conf["t_len"]-1,conf["patch"]["center_pixel"],conf["patch"]["center_pixel"]])
			if debug>=2: print("train_labels[count]",train_labels[count])
			
			count=count+1
	else:
		train_ims=[]
	return data	
		
		#print(im_name)
		#print(im.shape)
	#print(im_names)
	
def im_store_npy(path,name,band_n,out_path,in_rgb=False,patches_extract_flag=True):
	im = load_landsat(path+name+"/",band_n)
	mask = cv2.imread(path+"labels/"+name[3]+".tif")
	print(im.shape)
	#np.save(out_path+name,im)
	if patches_extract_flag:
		patch_shape=(32,32,6)
		patches=np.squeeze(view_as_windows(im,patch_shape,step=16))
		patches_path=path+name+"/"+"patches/"
		print(patches_path)
		pathlib.Path(patches_path).mkdir(parents=True, exist_ok=True) 
		
		print(patches.shape)
		for i in range(patches.shape[0]):
			for k in range(patches.shape[1]):
				pass

				np.save(patches_path+"patch_"+str(i)+"_"+str(k)+".npy",patches[i,k,1,:,:])
		
	#if in_rgb:
#		rgb=im[:,:,0:3]
#		cv2.imwrite("rgb_"+name+".png")
# -----------------------------
# new added functions for pix2pix
def load_landsat(path,band_n):
	images = sorted(glob.glob(path + '*.tif'))
	band = cv2.imread(images[0],0)
	rows, cols = band.shape
	img = np.zeros((rows, cols, band_n), dtype='uint16')
	num_band = 0
	for im in images:
		if 'B8' not in im and 'QB' not in im:
			#band = load_tiff_image(im)
			print(im)
			band = cv2.imread(im,-1)
			#print(band.dtype)
			img[:, :, num_band] = band
			num_band += 1
	return img


def patches_extract(im,patch_size):

	return 0

conf={"band_n": 6, "t_len":9, "path": "../data/"}
conf["out_path"]=conf["path"]+"results/"
conf["in_npy_path"]=conf["path"]+"in_npy/"
conf["in_rgb_path"]=conf["path"]+"in_rgb/"
conf["patch"]={}
conf["patch"]={"size":32, "stride":16, "out_npy_path":conf["path"]+"patches_npy/"}
conf["patch"]["ims_path"]=conf["patch"]["out_npy_path"]+"patches_all"
conf["patch"]["labels_path"]=conf["patch"]["out_npy_path"]+"labels_all"
conf['patch']['center_pixel']=int(np.around(conf["patch"]["size"]/2))
print(conf)
if __name__ == "__main__":
	conf["utils_main_mode"]=3

	if conf["utils_main_mode"]==1:
		im_patches_npy_multitemporal_from_npy_from_folder_store(conf)
	elif conf["utils_main_mode"]==2:
		names=["im1_190800","im2_200900","im3_221000","im4_190201","im5_230301","im6_080401","im7_020501","im8_110601","im9_050701"]
		for name in names:
			im_store_npy(conf["path"],name,conf["band_n"],out_path=conf["in_npy_path"],in_rgb=True)
	elif conf["utils_main_mode"]==3:
		#aa=np.load(conf["patch"]["out_npy_path"]+"labels_all/patch_0_0_0.npy")
		#print(aa.shape)
		data=im_patches_npy_multitemporal_from_npy_from_folder_load(conf,1)
		#pass
