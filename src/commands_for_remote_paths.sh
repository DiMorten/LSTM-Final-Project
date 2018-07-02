export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
export PATH=~/anaconda3/bin:$PATH

source activate epyt


export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64;export PATH=~/anaconda3/bin:$PATH;source activate epyt;cd ~/Documents/Jorg/deep_learning/LSTM-Final-Project/src

#pip install Cython
#conda create -n epyt
#conda install tensorflow-gpu
#conda install scikit-learn
#conda install scikit-image
#conda install -c anaconda scikit-image 
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

python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"

python main.py -mm="ram" --debug=1 -po 27 -ts 10 -tnl 10000000 --batch_size=5 --filters=256 -pl=32 -m="smcnn_unet" -nap=16000

python main.py -mm="ram" --debug=1 -po 4 -ts 4 -tnl 1000000 --batch_size=2000 --filters=256 -pl=5 -m="smcnn_unet" -nap=160000

toy version:
python main.py -mm="ram" --debug=1 -po 1 -ts 0 -tnl 1000 -bs=1500 --batch_size=200 --filters=256
python main.py -mm="ram" --debug=1 -po 0 -bs=500 --filters=32 -m="smcnn"

python main.py -mm="ram" --debug=1 -po 0 -bs=500 --filters=32 -m="unet"
python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="unet" -pl=32 -nap=16000

python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="smcnn_unet" -pl=32 -nap=16000

# Pending: Use validation set from train set
# 

python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn" --class_n=9 -sc=False

Class balancing

Test: [@debug] stats["per_class_label_count"] = [     0. 873240.      0.      0. 166194. 141965. 174402. 435596. 247387.]

[ 1.  4.  5.  6.  7.  8.] [326256 135005 109385 188598 482812 239672]



[ 0.58075876,  0.06860779,  0.        ,  0.21212446,  1.        ,
        0.3488955 ]




[ 0.6326076 ,  0.93579704,  1.        ,  0.82499779,  0.5       , 0.74134727]

[0, 0.6326076 ,  0, 0, 0.93579704,  1.        ,  0.82499779,  0.5       , 0.74134727]



Results:

SMCNN: 
[@debug] stats["overall_accuracy"] = 0.839626607065964
[@debug] stats["average_accuracy"] = 0.8172330218068024
python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=200 --filters=256 -m="smcnn" --class_n=9 -sc=False

python main.py -mm="ram" --debug=1 -pl 5 -po 0 -ts 1 -tnl 10000000 -bs=1000 --batch_size=200 --filters=256 -m="smcnnlstm" --class_n=9 -sc=False


ConvLSTM:
[@debug] self.early_stop["best"] = 0.8355461461174106
[@debug] self.early_stop["best_aa"] = 0.8134161141264326
python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=200 --filters=256 -m="convlstm" 


[@debug] early_stop["best"]["metric1"] = 0.8388346080501958
[@debug] early_stop["best"]["metric2"] = 0.8177334733912126
[@debug] early_stop["best"]["metric3"] = [0.90358998 0.80523872 0.87033873 0.82101485 0.80454534 0.70167322]