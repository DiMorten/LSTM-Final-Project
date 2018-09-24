 
import numpy as np
import glob
import deb
def val_set_get(self,mode='stratified',validation_split=0.2):
		ram_data['train']['idx']=range(ram_data['train']['n'])
		clss_train_unique,clss_train_count=np.unique(ram_data['train']['labels_int'],return_counts=True)
		deb.prints(clss_train_count)
		ram_data['val']={'n':int(ram_data['train']['n']*validation_split)}
		
		#===== CHOOSE VAL IDX
		#mode='stratified'
		if mode=='random':
			ram_data['val']['idx']=np.random.choice(ram_data['train']['idx'],ram_data['val']['n'],replace=False)
			

			ram_data['val']['ims']=ram_data['train']['ims'][ram_data['val']['idx']]
			ram_data['val']['labels_int']=ram_data['train']['labels_int'][ram_data['val']['idx']]
		
		elif mode=='stratified':
			while True:
				ram_data['val']['idx']=np.random.choice(ram_data['train']['idx'],ram_data['val']['n'],replace=False)
				ram_data['val']['ims']=ram_data['train']['ims'][ram_data['val']['idx']]
				ram_data['val']['labels_int']=ram_data['train']['labels_int'][ram_data['val']['idx']]
		
				clss_val_unique,clss_val_count=np.unique(ram_data['val']['labels_int'],return_counts=True)
				
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

				ram_data['val']['idx']=np.random.choice(ram_data['train']['idx'],ram_data['val']['n'],replace=False)
				

				ram_data['val']['ims']=ram_data['train']['ims'][ram_data['val']['idx']]
				ram_data['val']['labels_int']=ram_data['train']['labels_int'][ram_data['val']['idx']]
				clss_val_unique,clss_val_count=np.unique(ram_data['val']['labels_int'].argmax(axis=3),return_counts=True)
						
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

		deb.prints(ram_data['val']['idx'].shape)

		
		deb.prints(ram_data['val']['ims'].shape)
		#deb.prints(data.patches['val']['labels_int'].shape)
		
		
		ram_data['train']['ims']=np.delete(ram_data['train']['ims'],ram_data['val']['idx'],axis=0)
		ram_data['train']['labels_int']=np.delete(ram_data['train']['labels_int'],ram_data['val']['idx'],axis=0)
		ram_data['train']['n']=ram_data['train']['ims'].shape[0]
		ram_data['val']['n']=ram_data['val']['ims'].shape[0]
		print("train",np.unique(ram_data['train']['labels_int'],return_counts=True))
		print("val",np.unique(ram_data['val']['labels_int'],return_counts=True))
		return ram_data


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

t_step=7
patch_size=9
band_n=2

labels=np.load('train_labels_int.npy')
deb.prints(np.unique(labels,return_counts=True))
counter=0

val_chunk_size=int(total_size*val_percentage)
print(val_chunk_size)
val_ims=[]
val_labels=[]

val_counter=0
for file in files:
	
	#buffr=np.load(file,mmap_mode='r')
	ram_data={'train':{},'val':{}}
	ram_data['train']['ims']=np.load(file)
	print(file)
	
	ram_data['train']['n']=ram_data['train']['ims'].shape[0]
	deb.prints(ram_data['train']['n'])
	ram_data['train']['labels_int']=labels[counter:counter+ram_data['train']['n']]
	deb.prints(ram_data['train']['ims'].shape)
	
	deb.prints(ram_data['train']['labels_int'].shape)
	ram_data=val_set_get(ram_data,mode='stratified',validation_split=0.15)

	print("train",np.unique(ram_data['train']['labels_int'],return_counts=True))
	print("val",np.unique(ram_data['val']['labels_int'],return_counts=True))
	
	val_ims.append(ram_data['val']['ims'])
	val_labels.append(ram_data['val']['labels_int'])
	
	counter+=ram_data['train']['n']
	val_counter+=val_chunk_size	
	print("HEre2")
	#print(buffr[1:16])
	#print("Buffr",buffr.shape,np.unique(buffr,return_counts=True))

print(np.unique(labels,return_counts=True))
print(np.unique(val_ims,return_counts=True))