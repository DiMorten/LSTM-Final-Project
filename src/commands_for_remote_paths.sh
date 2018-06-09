export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
export PATH=~/anaconda3/bin:$PATH

source activate py361


#pip install Cython
#conda install tensorflow-gpu
#conda install scikit-learn
#conda install scikit-image
#conda install -c conda-forge opencv 
#cp cuda/include/cudnn.h /usr/local/cuda-9.2/include


#overlap 30, test skip 5. goood. 1000 balance samples per class. 1500 balance samples per class better.
#tensorboard --logdir "../data/summaries/"



#python main.py -mm="ram" --debug=1 -po 1 -ts 0

# 78% oa:
#python main.py -mm="ram" --debug=2 -po 3 -ts 8 -tnl 1000000 -bs=5000 --batch_size=500
python main.py -mm="ram" --debug=2 -po 4 -ts 8 -tnl 1000000 -bs=5000 --batch_size=1000
