 
import numpy as np
import glob
import deb
import sys

def val_set_get(buffr,mode='stratified',validation_split=0.2):
	buffr['train']['idx']=range(buffr['train']['n'])
	clss_train_unique,clss_train_count=np.unique(buffr['train']['labels_int'],return_counts=True)
	deb.prints(clss_train_count)
	buffr['val']={'n':int(buffr['train']['n']*validation_split)}
	
	#===== CHOOSE VAL IDX
	#mode='stratified'
	if mode=='random':
		buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
		

		buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
		buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
	
	elif mode=='stratified':
		while True:
			buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
			buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
			buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
	
			clss_val_unique,clss_val_count=np.unique(buffr['val']['labels_int'],return_counts=True)
			
			if not np.array_equal(clss_train_unique,clss_val_unique):
				deb.prints(clss_train_unique)
				deb.prints(clss_val_unique)
				
				pass
			else:
				percentages=clss_val_count/clss_train_count
				deb.prints(percentages)
				#if np.any(percentages<0.1) or np.any(percentages>0.3):
				if np.any(percentages>0.23):
				
					pass
				else:
					break
	elif mode=='random_v2':
		while True:

			buffr['val']['idx']=np.random.choice(buffr['train']['idx'],buffr['val']['n'],replace=False)
			

			buffr['val']['ims']=buffr['train']['ims'][buffr['val']['idx']]
			buffr['val']['labels_int']=buffr['train']['labels_int'][buffr['val']['idx']]
			clss_val_unique,clss_val_count=np.unique(buffr['val']['labels_int'].argmax(axis=3),return_counts=True)
					
			deb.prints(clss_train_unique)
			deb.prints(clss_val_unique)

			deb.prints(clss_train_count)
			deb.prints(clss_val_count)

			clss_train_count_in_val=clss_train_count[np.isin(clss_train_unique,clss_val_unique)]
			percentages=clss_val_count/clss_train_count_in_val
			deb.prints(percentages)
			#if np.any(percentages<0.1) or np.any(percentages>0.3):
			if np.any(percentages>0.26):
				pass
			else:
				break				

	deb.prints(buffr['val']['idx'].shape)

	
	deb.prints(buffr['val']['ims'].shape)
	#deb.prints(data.patches['val']['labels_int'].shape)
	
	
	buffr['train']['ims']=np.delete(buffr['train']['ims'],buffr['val']['idx'],axis=0)
	buffr['train']['labels_int']=np.delete(buffr['train']['labels_int'],buffr['val']['idx'],axis=0)
	buffr['train']['n']=buffr['train']['ims'].shape[0]
	buffr['val']['n']=buffr['val']['ims'].shape[0]
	print("train",np.unique(buffr['train']['labels_int'],return_counts=True))
	print("val",np.unique(buffr['val']['labels_int'],return_counts=True))
	return buffr
def labels_onehot_get(labels,n_samples,class_n):
	out=np.zeros((n_samples,class_n))
	deb.prints(out.shape)
	deb.prints(labels.shape)
	out[np.arange(n_samples),labels.astype(np.int)]=1
	return out
def data_balance( data, samples_per_class,class_n,debug=1):
	fname=sys._getframe().f_code.co_name

	balance={}
	balance["unique"]={}
#	classes = range(0,self.conf["class_n"])
	classes,counts=np.unique(data["train"]["labels_int"],return_counts=True)
	print(classes,counts)
	num_total_samples=len(classes)*samples_per_class
	balance["out_labels"]=np.zeros(num_total_samples)
	deb.prints(num_total_samples)
	balance["out_data"]=np.zeros(num_total_samples)
	
	#balance["unique"]=dict(zip(unique, counts))
	#print(balance["unique"])
	k=0
	for clss in classes:
		deb.prints(clss,fname)
		balance["data"]=data["train"]["ims"][data["train"]["labels_int"]==clss]
		balance["labels_int"]=data["train"]["labels_int"][data["train"]["labels_int"]==clss]
		balance["num_samples"]=balance["data"].shape[0]
		if debug>=1: deb.prints(balance["data"].shape,fname)
		if debug>=2: 
			deb.prints(balance["labels_int"].shape,fname)
			deb.prints(np.unique(balance["labels_int"].shape),fname)
		if balance["num_samples"] > samples_per_class:
			replace=False
		else: 
			replace=True

		index = range(balance["labels_int"].shape[0])
		index = np.random.choice(index, samples_per_class, replace=replace)
		balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["labels_int"][index]
		balance["out_data"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["data"][index]

		k+=1
	idx = np.random.permutation(balance["out_labels"].shape[0])
	balance["out_data"] = balance["out_data"][idx]
	balance["out_labels"] = balance["out_labels"][idx]

	balance["labels"]=labels_onehot_get(balance["out_labels"],num_total_samples,class_n)
	#balance["labels"]=np.zeros((num_total_samples,self.conf["class_n"]))
	#balance["labels"][np.arange(num_total_samples),balance["out_labels"].astype(np.int)]=1
	if debug>=1: deb.prints(np.unique(balance["out_labels"],return_counts=True),fname)
	return balance["out_data"],balance["out_labels"],balance["labels"]

inpath='train/'
val_percentage=0.15
files=sorted(glob.glob(inpath+'*.npy'), key=lambda x: x[11])
print(files)

# Get train total size
batch_sizes=[int(x[13:-4]) for x in files]
total_size=np.sum(np.asarray(batch_sizes))
print(batch_sizes,total_size)
"""
for file in files:
	#buffr=np.load(file,mmap_mode='r')
	buffr=np.load(file)
	
	print("HEre")
	#print(buffr[1:16])
	#print("Buffr",buffr.shape,np.unique(buffr,return_counts=True))
"""
#print(files[-1])
#buffr=np.load(files[-1])
#buffr=buffr[0:222054]
#np.save(files[-1],buffr)

dataset='cv'
if dataset=='cv':
	t_step=7
	patch_size=9
	band_n=2

labels=np.load('train_labels_int.npy')
deb.prints(np.unique(labels,return_counts=True))

val_chunk_size=int(total_size*val_percentage)
print(val_chunk_size)
val_ims=[]
val_labels=[]

val_counter=0
train_labels=[]

# === here save ims labels

data={"train":{},"val":{}}
data['train']['labels_int']=labels
data['train']['ims']=np.arange(labels.shape[0])
data['train']['n']=data['train']['labels_int'].shape[0]
data=val_set_get(data,mode='stratified',validation_split=0.15)

data['val']['idxs']=np.sort(data['val']['ims'])
print(data['val']['ims'][0:10])

# Count after validation split
data['val']['n']=data['val']['labels_int'].shape[0]

# Balancing
class_n=np.unique(data['train']['labels_int']).shape[0]
data['train']['idxs'],data['train']['labels_int'], \
data['train']['labels'] = data_balance(data,50000,class_n=class_n)

data['train']['n']=data['train']['labels_int'].shape[0]

data['train']['idxs']=data['train']['idxs'].astype(np.int)
data['val']['idxs']=data['val']['idxs'].astype(np.int)

deb.prints(data['train']['idxs'].dtype)
deb.prints(data['train']['idxs'][0:15])
deb.prints(np.unique(data['train']['labels_int'],return_counts=True))


counter=0


data['val']['ims']=np.zeros((data['val']['n'],t_step,
	patch_size,patch_size,band_n))
data['train']['ims']=np.zeros((data['train']['n'],t_step,
	patch_size,patch_size,band_n))

train_counter=0
for file in files:
	
	#buffr=np.load(file,mmap_mode='r')
	buffr={'train':{},'val':{}}
	buffr['train']['ims']=np.load(file)
	print(file)
	
	buffr['train']['n']=buffr['train']['ims'].shape[0]
	deb.prints(buffr['train']['n'])
	buffr['train']['labels_int']=labels[counter:counter+buffr['train']['n']]
	deb.prints(buffr['train']['ims'].shape)
	deb.prints(buffr['train']['labels_int'].shape)
	
	# Append val values
	buffr['val_idxs']=data['val']['idxs'][(data['val']['idxs'] \
		>=counter) & (data['val']['idxs']<counter+buffr['train']['n'])]
	
	# Val indices . subtract counter to access them from 0
	buffr['val_idxs']=buffr['val_idxs']-counter
	
	buffr['val_n']=buffr['val_idxs'].shape[0]
	data['val']['ims'][val_counter:val_counter+buffr['val_n']] = \
		buffr['train']['ims'][buffr['val_idxs']]

	# Append train values
	buffr['train_idxs']=data['train']['idxs'][(data['train']['idxs'] \
		>=counter) & (data['train']['idxs']<counter+buffr['train']['n'])]
	buffr['train_idxs']=buffr['train_idxs']-counter
	buffr['train_n']=buffr['train_idxs'].shape[0]
	data['train']['ims'][train_counter:train_counter+buffr['train_n']] = \
		buffr['train']['ims'][buffr['train_idxs']]

	counter+=buffr['train']['n']
	val_counter+=buffr['val_n']
	train_counter+=buffr['train_n']
	print("HEre2")


np.save('train_ims.npy',data['train']['ims'])
np.save('train_labels_int',data['train']['labels_int'])
np.save('val_ims.npy',data['val']['ims'])
np.save('val_labels_int',data['val']['labels_int'])

	"""
	buffr=val_set_get(buffr,mode='stratified',validation_split=0.15)

	print("train",np.unique(buffr['train']['labels_int'],return_counts=True))
	print("val",np.unique(buffr['val']['labels_int'],return_counts=True))
	
	val_ims.append(buffr['val']['ims'])
	val_labels.append(buffr['val']['labels_int'])
	
	#np.save(file,buffr['train']['ims'])
	
	train_labels.append(buffr['train']['labels_int'])

	#print(buffr[1:16])
	#print("Buffr",buffr.shape,np.unique(buffr,return_counts=True))
	"""
"""
print(len(val_labels,return_counts=True))
val_labels=np.concatenate(np.asarray(val_labels))
print(len(val_ims,return_counts=True))
val_ims=np.concatenate(np.asarray(val_ims))

print(len(train_labels,return_counts=True))
train_labels=np.concatenate(np.asarray(train_labels))


#====== BALANCING TFOOIT

#labels_unique,labels_count=np.unique(labels,return_counts=True)

#

#balanced_idxs,balanced_labels_int,balanced_labels=data_balance(data)

"""