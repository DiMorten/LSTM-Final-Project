import numpy as np
import cv2
import deb

mask=cv2.imread("../hn_data/TrainTestMask.tif",-1).astype(np.uint8)
#mask=cv2.imread("../cv_data/old_TrainTestMask.tif",0)

mask_test=mask.copy()
mask_test[mask_test==1]=0
mask_test[mask_test==2]=1
deb.prints("Mask test",np.count_nonzero(mask_test))


deb.prints(mask.shape)
mask_train=mask.copy()
mask_train[mask_train==2]=0




deb.prints(np.max(mask_train))
deb.prints(mask_train.dtype)
mask_train*=255
mask_patch_train=cv2.imread('mask_train.png',0)
deb.prints(mask_patch_train.shape)
deb.prints(mask_patch_train.dtype)
deb.prints(np.max(mask_patch_train))
dif=mask_train-mask_patch_train
deb.prints(np.max(dif))
deb.prints(np.unique(dif,return_counts=True))
deb.prints(np.count_nonzero(dif))
#cv2.imwrite('dif.png',dif)

#=== Test test
deb.prints("Test test")
mask_test=mask.copy()
mask_test[mask_test==1]=0
mask_test[mask_test==2]=1

deb.prints(np.max(mask_test))
deb.prints(mask_test.dtype)
mask_test*=255
mask_patch_test=cv2.imread('mask_test.png',0)
deb.prints(mask_patch_test.shape)
deb.prints(mask_patch_test.dtype)
deb.prints(np.max(mask_patch_test))
dif=cv2.absdiff(mask_test,mask_patch_test)
deb.prints(np.max(dif))
deb.prints(np.unique(dif,return_counts=True))

deb.prints(np.count_nonzero(dif))
cv2.imwrite('dif_test.png',dif)

#==== Test 2
deb.prints("Test label")
mask[mask==2]=1
mask_valid=mask.copy()
mask_valid[mask>0]=1
valid=cv2.imread('../hn_data/labels/24.tif',0)
labels=valid.copy()
deb.prints(np.unique(valid,return_counts=True))
valid[valid>0]=1
dif=cv2.absdiff(mask,valid)
deb.prints(np.max(dif))
deb.prints(np.count_nonzero(dif))
#cv2.imwrite('dif.png',dif)

labels_train=labels[mask_train==255]
deb.prints(labels_train.shape)
deb.prints(np.unique(labels_train,return_counts=True))