import numpy as np
import cv2

mask=cv2.imread("../cv_data/TrainTestMask.tif",0)
print(mask.shape)
mask_train=mask.copy()
mask_train[mask_train==2]=0
print(np.max(mask_train))
print(mask_train.dtype)
mask_train*=255
mask_patch_train=cv2.imread('mask_train.png',0)
print(mask_patch_train.shape)
print(mask_patch_train.dtype)
print(np.max(mask_patch_train))
dif=cv2.absdiff(mask_train,mask_patch_train)
print(np.max(dif))
print(np.count_nonzero(dif))
cv2.imwrite('dif.png',dif)

#==== Test 2

mask[mask==2]=1
valid=cv2.imread('../cv_data/labels/7.tif',0)

valid[valid>0]=1
dif=cv2.absdiff(mask_train,mask_patch_train)
print(np.max(dif))
print(np.count_nonzero(dif))
cv2.imwrite('dif.png',dif)


