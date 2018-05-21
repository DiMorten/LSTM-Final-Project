 
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

# -----------------------------
# new added functions for pix2pix
def load_landsat(path):
    images = sorted(glob.glob(path + '*.tif'))
    band = load_tiff_image(images[0])
    rows, cols = band.shape
    img = np.zeros((rows, cols, 7), dtype='float32')
    num_band = 0
    for im in images:
        if 'B8' not in im and 'QB' not in im:
            band = load_tiff_image(im)
            img[:, :, num_band] = band
            num_band += 1
        if 'QB' in im:
            cloud_mask = load_tiff_image(im)
            cloud_mask[cloud_mask != 0] = 1
    return img, cloud_mask