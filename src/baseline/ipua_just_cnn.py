import numpy as np
from csv import reader
import glob
#from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import preprocessing as pp
import timeit
import scipy.io
import random as rng
#from autoencoderSparse import learnFeatures
#from im2col import im2col
# import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers
import cv2
from skimage.util import view_as_windows

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_image(patch):
    # Read Mask Image
    #gdal_header = gdal.Open(patch)
    #img = gdal_header.ReadAsArray()
    img=cv2.imread(patch,0)
    DIM = img.shape
    img = img.reshape(DIM[0]*DIM[1])
    return img


def stack_images(start, images_list, seq):

    img = np.load(images_list[start-1])
    img = np.transpose(img,(2,0,1))
    print("img",img.shape)
    print(images_list[start-1])
    bands, rows, cols = img.shape
    img = img.reshape(bands, rows*cols)
    ndvi = (img[3] - img[2])/(img[3] + img[2])
    img = np.vstack((img, ndvi))
    bands += 1
    stack = np.zeros((seq*bands, rows*cols), dtype='float32')
    stack[0:bands] = img
    img = []
    for i in range(1, seq):
        print(images_list[start + i-1])
        img = np.load(images_list[start + i-1])
        img = img.reshape(bands-1, rows*cols)
        ndvi = (img[3] - img[2])/(img[3] + img[2])
        img = np.vstack((img, ndvi))
        stack[bands*i:bands*i+bands] = img

    return stack


def load_data(DIM, image2classify, start, images_list, labels_list, mask, seq):

    #  Load Image Labels
    # print(labels_list)
    labels = load_image(labels_list[image2classify])
    data = stack_images(start-seq+1, images_list, seq)
    print("stack",data.shape)
    return data, labels


def balance_data(data, labels, samples_per_class):

    shape = data.shape
    if len(shape) > 2:
        data = data.reshape(shape[0], shape[1]*shape[2]*shape[3])
    classes = np.unique(labels)
    print(classes)
    num_total_samples = len(classes)*samples_per_class
    out_labels = np.zeros((num_total_samples), dtype='float32')
    out_data = np.zeros((num_total_samples, data.shape[1]), dtype='float32')

    k = 0
    for clss in classes:
        clss_labels = labels[labels == clss]
        clss_data = data[labels == clss]
        num_samples = len(clss_labels)
        if num_samples > samples_per_class:
            # Choose samples randomly
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=False)
            out_labels[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]

        else:
            # do oversampling
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=True)
            out_labels[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]
        k += 1
    # Permute samples randomly
    idx = np.random.permutation(out_data.shape[0])
    out_data = out_data[idx]
    out_labels = out_labels[idx]

    if len(shape) > 2:
        out_data = out_data.reshape(out_data.shape[0], shape[1], shape[2], shape[3])

    return out_data, out_labels


def extract_patches_randomly(data, DIM, ksize=3, numPatches=500000):
    bands, num_pixels = data.shape
    data = data.reshape(bands, DIM[0], DIM[1])

    N = bands * ksize**2
    patches = np.zeros((numPatches, N), dtype='float32')
    for i in range(numPatches):
        # if (np.mod(i, 10000) == 0):
            # print('Extracting patch:', i, numPatches)
        r = rng.randint(0, DIM[0] - ksize)
        c = rng.randint(0, DIM[1] - ksize)
        patch = data[:, r:r + ksize, c:c + ksize]
        patches[i] = patch.reshape(N)

    return patches


# def extract_patches(img, ksize=3):
#     bands, rows, cols = img.shape

#     padding = np.zeros(shape=(bands, rows + 2*(ksize/2), cols + 2*(ksize/2)),
#                        dtype='float32')
#     padding[:, ksize/2:padding.shape[1] - ksize/2, ksize/2:padding.shape[2] - ksize/2] = img
#     patches = im2col(padding[0],  [ksize, ksize])
#     for channel in range(1, bands):
#         patches = np.vstack((patches, im2col(padding[channel], [ksize, ksize])))
#     patches = patches.T

#     return patches

def extract_patches(img, ksize=3):
    img=np.transpose(img,(1,2,0)) # Switch to channel last
    padding = np.zeros(shape=(img.shape[0] + 2*int(ksize/2),
                              img.shape[1] + 2*int(ksize/2),
                              img.shape[2]), dtype=img.dtype)
    ksize_half=int(ksize/2)
    #padding[ksize/2:padding.shape[0] - ksize/2, ksize/2:padding.shape[1] - ksize/2, :] = img
    padding[ksize_half:padding.shape[0] - ksize_half, ksize_half:padding.shape[1] - ksize_half, :] = img
    
    kernel = (ksize, ksize, img.shape[2])
    print("kernel",kernel)
    subimgs = view_as_windows(padding, kernel)
    subimgs = np.squeeze(subimgs)
    print("subimgs",subimgs.shape)
    subimgs = np.transpose(subimgs,(0,1,4,2,3)) # Go back to channel first
    subimgs = np.reshape(subimgs,(img.shape[0]*img.shape[1],subimgs.shape[2],subimgs.shape[3],subimgs.shape[4]))
    subimgs = np.reshape(subimgs,(subimgs.shape[0],subimgs.shape[1]*subimgs.shape[2]*subimgs.shape[3]))


    #subimgs = np.reshape(subimgs,(img.shape[0]*img.shape[1],subimgs.shape[2]*subimgs.shape[3]*subimgs.shape[4]))
    print("subimgs",subimgs.shape)
    
    return subimgs

def extract_subimages(img, mask, DIM, ksize=5):
    bands, num_pixels = img.shape
    print(img.shape,bands,num_pixels)
    img = img.reshape(bands, DIM[0], DIM[1])

    subimgs = extract_patches(img, ksize)
    trn_subimgs = subimgs[mask == 1]
    tst_subimgs = subimgs[mask == 2]
    # Normalization sub images
    scaler = pp.StandardScaler().fit(trn_subimgs)
    trn_subimgs = scaler.transform(trn_subimgs)
    tst_subimgs = scaler.transform(tst_subimgs)
    trn_subimgs = trn_subimgs.reshape(len(trn_subimgs), bands, ksize, ksize)
    tst_subimgs = tst_subimgs.reshape(len(tst_subimgs), bands, ksize, ksize)

    return trn_subimgs, tst_subimgs


def extract_features(stack, DIM, weights, bias, scaler, ksize=3):
    bands, num_pixels = stack.shape
    stack = stack.reshape(bands, DIM[0], DIM[1])
    patches = extract_patches(stack, ksize)
    patches = scaler.transform(patches)
    features = sigmoid(np.matmul(weights, (patches).T).T + bias)

    return features


if __name__ == '__main__':
    np.core.arrayprint._line_width = 160  # terminal width
    #root_directory = '/home/jose/Drive/PUC/WorkPlace/IpuaCodes/'
    #root_directory = '/home/jorg/Documents/Master/scnd_semester/neural_nets/final_project/repo2/LSTM-Final-Project/src/baseline/'
    root_directory='/home/lvc/Documents/Jorg/deep_learning/LSTM-Final-Project/src/baseline/'
    #images_directory = '/mnt/Data/DataBases/RS/Ipua/'
    images_directory = '../../data/'


#    root_directory = '/home/jose/Workplace/Experiments/Ipua/Codes/'
#    images_directory = '/home/jose/DataBases/RS/Ipua/npy_imgs/'
    images_list = glob.glob(images_directory + 'in_npy2/' '*.npy')
    print(images_list)
    images_list.sort(key=lambda f: int(f[21]))  # Sort lists
    labels_list = glob.glob(images_directory + 'labels/' '*.tif')
    print(labels_list)
    labels_list.sort(key=lambda f: int(f[18]))  # Sort lists
    mask_patch = [images_directory + 'TrainTestMask.tif']
    print(mask_patch)
    output_folder = 'results/'

    filename_accuracy = 'Report_ipua_cnn.txt'
    file_ouput = open(filename_accuracy, 'w')

    DIM = [948, 1068]
    n_trees = 250
    max_depth = 25
    samples_per_class = 5000
    batch_size = 30
    depth1 = 64
    sub_size = 7
    k_size = 3

    mask = load_image(mask_patch[0])
    print("mask.shape",mask.shape)
    num_images = 9
    start = 4
    end = 9
    # range(end, end-1, -1)
    for im in range(end, start-1, -1)[0:1]:
#        range(im-1, start-2, -1)
        sequence = range(im-start+1, 0, -1)
        seq_length = 5
        for seq in sequence[0:-2]:
            image2classify = np.array(range(im-1, im-1 - 1, -1))
            seq_length += 1
            # range(im-1, im-1 + seq_length)
            for img_index in image2classify[0:1]:
                file_name = images_list[img_index][len(images_directory):
                                                   len(images_list[img_index])-4]
                print(file_name)
                stack, labels = load_data(DIM,
                                          img_index,
                                          im,
                                          images_list,
                                          labels_list,
                                          mask,
                                          seq_length)

#                depth = int(9*stack.shape[0]*2.0)
                depth = depth1

                print("stack",stack.shape)
                print("labels",labels.shape)
                
                trn_subimgs, tst_subimgs = extract_subimages(stack,
                                                             mask,
                                                             DIM,
                                                             ksize=sub_size)

                num_classes = len(np.unique(labels))-1  # Background
                # create mapping of characters to integers (0-25) and the reverse
                classes = np.unique(labels)
                labels2new_labels = dict((c, i) for i, c in enumerate(classes))
                new_labels2labels = dict((i, c) for i, c in enumerate(classes))
                new_labels = labels.copy()
                for i in range(len(classes)):
                    new_labels[labels == classes[i]] = labels2new_labels[classes[i]]

                trn_labels = new_labels[mask == 1]
                tst_labels = new_labels[mask == 2]

#                class_weighting = get_class_weights(new_labels[new_labels!=0], 1)
                print("trn_subimgs.shape",trn_subimgs.shape)
                count,unique=np.unique(trn_labels,return_counts=True)
                print("train count,unique",count,unique)
                trn_subimgs, trn_labels = balance_data(trn_subimgs,
                                                       trn_labels,
                                                       samples_per_class)


                # convert class vectors to binary class matrices
                trn_labels = keras.utils.to_categorical(trn_labels, num_classes+1)
                tst_labels = keras.utils.to_categorical(tst_labels, num_classes+1)

                epochs = 100
                input_shape = (7*seq_length, sub_size, sub_size)
#MaxPooling2D, AveragePooling2D
                model = Sequential()
                model.add(Conv2D(265,
                                 kernel_size=(k_size, k_size),
                                 activation='relu',
                                 padding="valid",
#                                 kernel_regularizer=regularizers.l2(10e-5),
                                 input_shape=input_shape))
#                model.add(Dropout(0.5))
#                model.add(Conv2D(256, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                model.add(Dropout(0.5))
                model.add(Dense(256, activation='relu'))
#                model.add(Dropout(0.5))
                model.add(Dense(num_classes+1, activation='softmax'))

                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adadelta(),
                              metrics=['accuracy'])

                model.fit(trn_subimgs, trn_labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
#                          class_weight=class_weighting,
                          validation_data=(tst_subimgs,
                                           tst_labels)
                          )
                score = model.evaluate(tst_subimgs, tst_labels, verbose=0)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                predictions = model.predict_classes(tst_subimgs)
                # Back to original labels
                classes = np.unique(predictions)
                new_predictions = predictions.copy()
                for i in range(len(classes)):
                    new_predictions[predictions == classes[i]] = new_labels2labels[classes[i]]

#                 accuracy
                accuracy = accuracy_score(labels[mask == 2], new_predictions)
                print('accuracy', 100*accuracy)
                conf_matrix = confusion_matrix(labels[mask == 2],
                                               new_predictions)
                F1_Score = f1_score(labels[mask == 2], new_predictions,
                                    average=None)
                print(conf_matrix)
                print(100 * F1_Score)
                print(np.sum(100 * F1_Score))
#                file_ouput.write('Image --> ' + file_name + '\n')
#                file_ouput.write('Sequence length --> ' + str(seq_length) + '\n')
#                file_ouput.write('F1 Score' + '\n')
#                file_ouput.write(str(100 * F1_Score) + '\n')
#                file_ouput.write('F1 Score =' + str(np.sum(100 * F1_Score)) + '\n')
#                # Save data
#                mat_realpred = np.array((labels[mask == 2], new_predictions))
#                file_output_name = output_folder + file_name + '_' + str(im) + '_' + str(seq_length) + '.mat'
#                scipy.io.savemat(file_output_name, mdict={'mat_real_pred': mat_realpred})
    file_ouput.close()
