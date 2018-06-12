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
python main.py -mm="ram" --debug=1 -po 4 -ts 8 -tnl 1000000 -bs=20000 --batch_size=3500

81%:
python main.py -mm="ram" --debug=1 -po 4 -ts 4 -tnl 1000000 -bs=20000 --batch_size=2000

python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256



toy version:
python main.py -mm="ram" --debug=1 -po 1 -ts 0 -tnl 1000 -bs=1500 --batch_size=200 --filters=256
python main.py -mm="ram" --debug=1 -po 0 -bs=500 --filters=32

python main.py -mm="ram" --debug=1 -po 0 -bs=500 --filters=32 -m="unet"
python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="unet" -pl=32 -nap=16000

python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="smcnn_unet" -pl=32 -nap=16000

# Pending: Use validation set from train set
# 



Class balancing

Test: [@debug] stats["per_class_label_count"] = [     0. 873240.      0.      0. 166194. 141965. 174402. 435596. 247387.]

[ 1.  4.  5.  6.  7.  8.] [326256 135005 109385 188598 482812 239672]