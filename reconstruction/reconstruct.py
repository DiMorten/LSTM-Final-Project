 
import numpy as np
import cv2

cols=np.load('locations_col.npy')
rows=np.load('locations_row.npy')
label_checker=np.load('locations_label.npy')

folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/results/convlstm_16_300_nompool/'
mask_path='../hn_data/TrainTestMask.tif'

labels=np.load(folder+'labels.npy')
predictions=np.load(folder+'predicted.npy')



print(cols.shape)

print(label_checker.shape,labels.dtype)
print(np.unique(label_checker,return_counts=True))


print(labels.shape,labels.dtype)
print(np.unique(labels,return_counts=True))

print(predictions.shape,predictions.dtype)
print(np.unique(predictions,return_counts=True))

# Transform back

labels=labels+1
predictions=predictions+1


print(np.all(labels==label_checker))

mask=cv2.imread(mask_path,-1)
print("Mask shape",mask.shape)

label_rebuilt=np.zeros_like(mask)
prediction_rebuilt=np.zeros_like(mask)
for row,col,label,prediction in zip(rows,cols,labels,predictions):
	label_rebuilt[row,col]=label
	prediction_rebuilt[row,col]=prediction

print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


