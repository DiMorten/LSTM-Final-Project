import numpy as np
import cv2

mask=cv2.imread("../cv_data/TrainTestMask.tif",0)
#mask=cv2.imread("../cv_data/old_TrainTestMask.tif",0)

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

#=== Test test
print("Test test")
mask_test=mask.copy()
mask_test[mask_test==1]=0
mask_test[mask_test==2]=1

print(np.max(mask_test))
print(mask_test.dtype)
mask_test*=255
mask_patch_test=cv2.imread('mask_test.png',0)
print(mask_patch_test.shape)
print(mask_patch_test.dtype)
print(np.max(mask_patch_test))
dif=cv2.absdiff(mask_test,mask_patch_test)
print(np.max(dif))
print(np.count_nonzero(dif))
#cv2.imwrite('dif.png',dif)

#==== Test 2
print("Test label")
mask[mask==2]=1
mask_valid=mask.copy()
mask_valid[mask>0]=1
valid=cv2.imread('../cv_data/labels/7.tif',0)

valid[valid>0]=1
dif=cv2.absdiff(mask,valid)
print(np.max(dif))
print(np.count_nonzero(dif))
#cv2.imwrite('dif.png',dif)


