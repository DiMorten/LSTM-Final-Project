
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
from model import (conv_lstm,Conv3DMultitemp,UNet,SMCNN,SMCNNlstm)

#import conf
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='phase')
parser.add_argument('--dataset_name', dest='dataset_name', default='20160419', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--timesteps', dest='timesteps', type=int, default=utils.conf["t_len"], help='# timesteps used to train')
parser.add_argument('-pl','--patch_len', dest='patch_len', type=int, default=5, help='# timesteps used to train')
parser.add_argument('--kernel', dest='kernel', type=int, default=[3,3], help='# timesteps used to train')
parser.add_argument('--channels', dest='channels', type=int, default=6, help='# timesteps used to train')
parser.add_argument('--filters', dest='filters', type=int, default=32, help='# timesteps used to train')
parser.add_argument('--n_classes', dest='n_classes', type=int, default=9, help='# timesteps used to train')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('-m','--model', dest='model', default='convlstm', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default=utils.conf["summaries_path"], help='models are saved here')

parser.add_argument('--debug', type=int, default=1, help='Debug')
parser.add_argument('-po','--patch_overlap', dest='patch_overlap', type=int, default=0, help='Debug')
parser.add_argument('--im_size', dest='im_size', type=int, default=(948,1068), help='Debug')
parser.add_argument('--band_n', dest='band_n', type=int, default=6, help='Debug')
parser.add_argument('--t_len', dest='t_len', type=int, default=6, help='Debug')
parser.add_argument('--path', dest='path', default="../data/", help='Data path')
parser.add_argument('--class_n', dest='class_n', type=int, default=9, help='Class number')
parser.add_argument('--pc_mode', dest='pc_mode', default="local", help="Class number. 'local' or 'remote'")
parser.add_argument('-tnl','--test_n_limit', dest='test_n_limit',type=int, default=1000, help="Class number. 'local' or 'remote'")
parser.add_argument('-mm','--memory_mode', dest='memory_mode',default="hdd", help="Class number. 'local' or 'remote'")
parser.add_argument('-bs','--balance_samples_per_class', dest='balance_samples_per_class',type=int,default=None, help="Class number. 'local' or 'remote'")
parser.add_argument('-ts','--test_get_stride', dest='test_get_stride',type=int,default=8, help="Class number. 'local' or 'remote'")
parser.add_argument('-nap','--n_apriori', dest='n_apriori',type=int,default=1000000, help="Class number. 'local' or 'remote'")
args = parser.parse_args()
np.set_printoptions(suppress=True)
if args.model=='unet':
    label_type='semantic'
else:
    label_type='one_hot'


"""
if args.memory_mode=="hdd":
    with open(utils.conf["path"]+'data.pkl', 'rb') as handle: dataset=pickle.load(handle)
    deb.prints(dataset["train"]["ims"].shape)
    deb.prints(dataset["train"]["labels_onehot"].shape)
    deb.prints(dataset["test"]["ims"].shape)
    deb.prints(dataset["test"]["labels_onehot"].shape)
    args.train_size = dataset["train"]["ims"].shape[0]
"""
def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if label_type=='one_hot':
        data=utils.DataOneHot(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
                                band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
                                test_n_limit=args.test_n_limit,memory_mode=args.memory_mode, \
                                balance_samples_per_class=args.balance_samples_per_class, test_get_stride=args.test_get_stride, \
                                n_apriori=args.n_apriori,patch_length=args.patch_len)
    elif label_type=='semantic':
        data=utils.DataSemantic(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
                                band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
                                test_n_limit=args.test_n_limit,memory_mode=args.memory_mode, \
                                balance_samples_per_class=args.balance_samples_per_class, test_get_stride=args.test_get_stride, \
                                n_apriori=args.n_apriori,patch_length=args.patch_len)


    if args.memory_mode=="ram":
        data.create()
        deb.prints(data.ram_data["train"]["ims"].shape)
    with tf.Session() as sess:
        if args.model=='convlstm':
            model = conv_lstm(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                            timesteps=args.timesteps, patch_len=args.patch_len,
                            kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                            checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,data=data.ram_data,conf=data.conf, debug=args.debug)
        elif args.model=='conv3d':
            model = Conv3DMultitemp(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                            timesteps=args.timesteps, patch_len=args.patch_len,
                            kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                            checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,data=data.ram_data, debug=args.debug)
        elif args.model=='unet':
            model = UNet(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                            timesteps=args.timesteps, patch_len=args.patch_len,
                            kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                            checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,data=data.ram_data, debug=args.debug)
        elif args.model=='smcnn':
            model = SMCNN(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                            timesteps=args.timesteps, patch_len=args.patch_len,
                            kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                            checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,data=data.ram_data, debug=args.debug)
        elif args.model=='smcnnlstm':
            model = SMCNNlstm(sess, batch_size=args.batch_size, epoch=args.epoch, train_size=args.train_size,
                            timesteps=args.timesteps, patch_len=args.patch_len,
                            kernel=args.kernel, channels=args.channels, filters=args.filters, n_classes=args.n_classes,
                            checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,data=data.ram_data, debug=args.debug)
        if args.phase == 'train':
            model.train(args)
        
        elif args.phase == 'test':
            model.test(args)
        """
        elif args.phase == 'generate_image':
            model.generate_image(args)
        elif args.phase == 'create_dataset':
            model.create_dataset(args)
        """
        #else:
        #    print ('...')


if __name__ == '__main__':
    tf.app.run()



