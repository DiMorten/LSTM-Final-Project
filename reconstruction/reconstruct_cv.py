 
import numpy as np
import cv2
#locations_folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/batched_rnn_src/'
cols=np.load('cv/locations_col.npy')
rows=np.load('cv/locations_row.npy')
label_checker=np.load('cv/locations_label.npy')

#folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/results/convlstm_16_300_nompool/'
folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/buffer/seq1/15/convlstm/'



# == normy3

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3/convlstm/seq1/'


mask_path='../cv_data/TrainTestMask.tif'

labels=np.load(folder+'labels.npy').argmax(axis=1)
predictions=np.load(folder+'predicted.npy').argmax(axis=1)



print(cols.shape)



# Transform back

"""
correspondence=np.array([0,1],
						[1,2],
						[2,3],
						[3,4],
						[4,6],
						[5,7],
						[6,8],
						[7,9],
						[8,10],
						[9,11])
"""
correspondence=np.array([ 1 , 2 , 3 , 4 , 6 , 7,  8 , 9 ,10 ,11])

print(np.all(labels==label_checker))

print(label_checker.shape,labels.dtype)
print(np.unique(label_checker,return_counts=True))


print("Labels",labels.shape,labels.dtype)
print(np.unique(labels,return_counts=True))

print(predictions.shape,predictions.dtype)
print(np.unique(predictions,return_counts=True))

labels_tmp=labels.copy()
predictions_tmp=predictions.copy()


for idx in range(correspondence.shape[0]):
	labels_tmp[labels==idx]=correspondence[idx]
	predictions_tmp[predictions==idx]=correspondence[idx]

labels=labels_tmp.copy()
predictions=predictions_tmp.copy()
#print(np.all(labels==label_checker))


print("Labels",labels.shape,labels.dtype)
print(np.unique(labels,return_counts=True))

print(predictions.shape,predictions.dtype)
print(np.unique(predictions,return_counts=True))

#labels=labels+1
#predictions=predictions+1



mask=cv2.imread(mask_path,-1)
print("Mask shape",mask.shape)

label_rebuilt=np.ones_like(mask).astype(np.uint8)*255
prediction_rebuilt=np.ones_like(mask).astype(np.uint8)*255
for row,col,label,prediction in zip(rows,cols,labels,predictions):
	label_rebuilt[row,col]=label
	prediction_rebuilt[row,col]=prediction

print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


# [ 1  2  3  4  6  7  8  9 10 11]


custom_colormap = np.array([[255, 146, 36],
                   [255, 255, 0],
                   [164, 164, 164],
                   [255, 62, 62],
                   [0, 0, 0],
                   [172, 89, 255],
                   [0, 166, 83],
                   [40, 255, 40],
                   [187, 122, 83],
                   [217, 64, 238],
                   [45, 150, 255]])
custom_colormap_tmp=custom_colormap.copy()

custom_colormap=custom_colormap[np.array([0,1,2,3,5,6,7,8,9,10])]
print(custom_colormap.shape)

#=== change to rgb
print("Gray",prediction_rebuilt.dtype)

prediction_rgb=cv2.cvtColor(prediction_rebuilt,cv2.COLOR_GRAY2RGB)
print("RGB",prediction_rgb.dtype)

print(prediction_rgb.shape)


for idx in range(custom_colormap.shape[0]):
	for chan in [0,1,2]:
		prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]

prediction_rgb=cv2.cvtColor(prediction_rgb,cv2.COLOR_BGR2RGB)

# == label]

label_rgb=cv2.cvtColor(label_rebuilt,cv2.COLOR_GRAY2RGB)
print("RGB",label_rgb.dtype)

print(label_rgb.shape)


for idx in range(custom_colormap.shape[0]):
	for chan in [0,1,2]:
		label_rgb[:,:,chan][label_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]

label_rgb=cv2.cvtColor(label_rgb,cv2.COLOR_BGR2RGB)


cv2.imwrite("reconstruction_normy3.png",prediction_rgb)
cv2.imwrite("label_normy3.png",label_rgb)

print(prediction_rgb[0,0,:])