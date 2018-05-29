 
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


def im_store_multitemporal_patches_npy_from_npy(conf,names):
	out_path={"patches":conf["patch"]["out_npy_path"]+"patches_all/","labels":conf["patch"]["out_npy_path"]+"labels_all/"}
	patch_shape=(conf["patch"]["size"],conf["patch"]["size"],conf["band_n"])
	label_shape=(conf["patch"]["size"],conf["patch"]["size"])
	pathlib.Path(out_path["patches"]).mkdir(parents=True, exist_ok=True) 
	pathlib.Path(out_path["labels"]).mkdir(parents=True, exist_ok=True) 
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

	print(out_path["patches"])
	print("patches",patches_all.shape)
	print("label_patches.s",label_patches_all.shape)
	count=0
	for i in range(patches.shape[0]):
		for k in range(patches.shape[1]):
			#pass
			np.save(out_path["patches"]+"patch_"+str(count)+"_"+str(i)+"_"+str(k)+".npy",patches_all[i,k,:,:,:,:])
			np.save(out_path["labels"]+"patch_"+str(count)+"_"+str(i)+"_"+str(k)+".npy",label_patches_all[i,k,:,:,:])
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

def im_store_multitemporal_patches_npy_from_npy_from_folder(conf):
	#ims["full"]=[]
	im_names=[]
	for i in range(1,10):
		im_name=glob.glob(conf["in_npy_path"]+'im'+str(i)+'*')[0]
		im_name=im_name[-14:-4]
		im_names.append(im_name)
		print(im_name)
	print(im_names)
	im_store_multitemporal_patches_npy_from_npy(conf,im_names)
		
		#ims["full"].append(np.load(im_name))
	#return ims




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
print(conf)
im_store_multitemporal_patches_npy_from_npy_from_folder(conf)
#im_store_patches_npy_from_npy_from_folder(conf)
#aa=cv2.imread("../data/labels/2.tif")
#print(aa.shape)
##if __name__ == "__main__":
##	names=["im1_190800","im2_200900","im3_221000","im4_190201","im5_230301","im6_080401","im7_020501","im8_110601","im9_050701"]
##	for name in names:
##		im_store_npy(conf["path"],name,conf["band_n"],out_path=conf["in_npy_path"],in_rgb=True)

#im = load_landsat(conf["path"]+"im1_190800/",conf["band_n"])
#im_rgb = im[:,:,0:3]
#print(im_rgb.shape)
#cv2.imwrite("im_rgb.png",im_rgb.astype(np.uint16))