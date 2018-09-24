 
import numpy as np
import glob

inpath='train/'

files=glob.glob(inpath+'*.npy')

for file in files:
	#buffr=np.load(file,mmap_mode='r')
	buffr=np.load(file)
	
	print("HEre")
	#print(buffr[1:16])
	#print("Buffr",buffr.shape,np.unique(buffr,return_counts=True))

