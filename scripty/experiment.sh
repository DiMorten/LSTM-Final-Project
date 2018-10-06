 

mode=1
if [ $mode -eq 0 ]; then
	cd ../hanover_rnn_src/
	python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=11000 --batch_size=1000 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="True" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/"

	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/lstm/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/lstm/label.npy
fi

if [ $mode -eq 1 ]; then
	cd ../src/
	echo "EXTRACTING FCN CAMPO VERDE PATCHES"
	python main.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=200 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof="custom" -nap=10000 -psv=True
fi
if [ $mode -eq 2 ]; then

	cd ../../../sbsr/fcn_model/keras_time_semantic_fcn/
	python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2
fi
