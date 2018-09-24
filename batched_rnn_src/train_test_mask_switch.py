 
 

from __future__ import division
import os
import math
import random
import time
import numpy as np
from time import gmtime, strftime
import glob
import tensorflow as tf
import numpy as np
from random import shuffle
import glob
import sys
import pickle

# Local
import utils
import deb
import cv2

path_in='../cv_data/TrainTestMask.tif'
path_out='../cv_data/TrainTestMask_switch.tif'
mask_switch=utils.mask_train_test_switch_from_path(path_in)
cv2.imwrite(path_out,mask_switch)