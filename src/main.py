
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
import tensorflow as tf
import numpy as np
from random import shuffle
#from tensorflow.contrib.rnn import ConvLSTMCell
import glob
import sys
import pickle
import argparse

# Local
import utils
import deb
from model import conv_lstm
#import conf
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='phase')
parser.add_argument('--dataset_name', dest='dataset_name', default='20160419', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--timesteps', dest='timesteps', default=utils.conf["t_len"], help='# timesteps used to train')
parser.add_argument('--shape', dest='shape', default=[5,5], help='# timesteps used to train')
parser.add_argument('--kernel', dest='kernel', default=[3,3], help='# timesteps used to train')
parser.add_argument('--channels', dest='channels', default=6, help='# timesteps used to train')
parser.add_argument('--filters', dest='filters', default=32, help='# timesteps used to train')
parser.add_argument('--n_classes', dest='n_classes', default=9, help='# timesteps used to train')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
args = parser.parse_args()
np.set_printoptions(suppress=True)

with open(utils.conf["path"]+'data.pkl', 'rb') as handle: dataset=pickle.load(handle)
deb.prints(dataset["train"]["ims"].shape)
deb.prints(dataset["train"]["labels_onehot"].shape)
deb.prints(dataset["test"]["ims"].shape)
deb.prints(dataset["test"]["labels_onehot"].shape)
args.train_size = dataset["train"]["ims"].shape[0]

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    with tf.Session() as sess:
        model = conv_lstm(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                        timesteps=args.timesteps, shape=args.shape,
                        kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                        checkpoint_dir=args.checkpoint_dir)

        if args.phase == 'train':
            model.train(args)
        """
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'generate_image':
            model.generate_image(args)
        elif args.phase == 'create_dataset':
            model.create_dataset(args)
        """
        #else:
        #    print ('...')


if __name__ == '__main__':
    tf.app.run()



