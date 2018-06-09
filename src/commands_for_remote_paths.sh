export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
export PATH=~/anaconda3/bin:$PATH

#source activate py361

conda activate epyt

#pip install Cython
#conda install tensorflow-gpu
#conda install scikit-learn
#conda install scikit-image
#conda install -c conda-forge opencv 
#cp cuda/include/cudnn.h /usr/local/cuda-9.2/include


#overlap 30, test skip 5. goood. 1000 balance samples per class. 1500 balance samples per class better.
#tensorboard --logdir "../data/summaries/"