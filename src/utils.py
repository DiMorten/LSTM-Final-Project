 
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
from osgeo import gdal
import glob
from skimage.transform import resize
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import cv2
#data_path="../data/"

def im_store_npy(path,name,band_n):
	im = load_landsat(path,band_n)
	np.save(name,im)
# -----------------------------
# new added functions for pix2pix
def load_landsat(path,band_n):
    images = sorted(glob.glob(path + '*.tif'))
    band = load_tiff_image(images[0])
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

def load_tiff_image(patch):
    # Read Mask Image
    #print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img

conf={"band_n": 6, "path": "../data/"}
names=["im1_190800","im2_200900","im3_221000","im4_190201","im5_230301","im6_080401","im7_020501","im8_110601","im9_050701"]
for name in names:
	im_store_npy(conf["path"]+name+"/",name,conf["band_n"])

#im = load_landsat(conf["path"]+"im1_190800/",conf["band_n"])
#im_rgb = im[:,:,0:3]
#print(im_rgb.shape)
#cv2.imwrite("im_rgb.png",im_rgb.astype(np.uint16))