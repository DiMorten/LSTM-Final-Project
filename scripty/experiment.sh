#folder="normy3_check2"
#folder_test="normy3"
KERAS_BACKEND=tensorflow

mode=9
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
	cd ../src/
	echo "EXTRACTING FCN CAMPO VERDE PATCHES SEQ 2"
	python main.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=200 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -tof="custom" -nap=10000 -psv=True
fi
if [ $mode -eq 3 ]; then

	cd ../../../sbsr/fcn_model/keras_time_semantic_fcn/
	python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2
fi
if [ $mode -eq 4 ]; then
	echo "STARTING HANOVER"
	cd ../hanover_rnn_src/
	python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=30000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/" 
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check/convlstm/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check/convlstm/labels.npy

	python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=30000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/" 
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check/lstm/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check/lstm/labels.npy


	cd ../batched_rnn_src/
	#echo "STARTING CONVLSTM FOR CAMPO VERDE"
	#python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/test_patches/seq1/"
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3/convlstm/seq1/predicted.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3/convlstm/seq1/labels.npy

	echo "SARTING LSTM FOR CAMPO VERDE"
	python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/test_patches/seq1/"
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/lstm/seq1/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/lstm/seq1/labels.npy

	python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=9 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/test_patches/seq2/"
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3/convlstm/seq2/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/convlstm/seq2/labels.npy



	python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=9 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/test_patches/seq2/"
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3/lstm/seq2/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/lstm/seq2/labels.npy

	
	# Here do FCN from all



	#python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=30000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/" 
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3/lstm/predicted_bs128.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3/lstm/labels_bs128.npy
fi
if [ $mode -eq 5 ]; then

	echo "STARTING HANOVER CONVLSTM"
	cd ../hanover_rnn_src/
	python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=15000 --batch_size=1000 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="True" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/" 
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/convlstm/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/convlstm/labels.npy
fi

if [ $mode -eq 6 ]; then

	echo "STARTING HANOVER CINVLSTM"
	cd ../hanover_rnn_src/
	python main.py -mm="ram" --debug=1 -pl 5 -po 4 -ts 0 -tnl 90 -bs=15000 --batch_size=1000 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=8 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/test_patches/" 
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/convlstm/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/convlstm/labels.npy
fi
if [ $mode -eq 7 ]; then


	python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/test_patches/seq1/"
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy4/convlstm/seq1/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy4/convlstm/seq1/labels.npy

fi
if [ $mode -eq 8 ]; then
	echo "SARTING LSTM FOR CAMPO VERDE"

	cd ../batched_rnn_src/

	#python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="True" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder_test}/test_patches/seq1/"
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/lstm/seq1/predicted.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/lstm/seq1/labels.npy

	#python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="lstm" --phase="train" -sc=True --class_n=9 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder_test}/test_patches/seq2/"
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/lstm/seq2/predicted.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/lstm/seq2/labels.npy

	#python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder_test}/test_patches/seq1/"
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq1/predicted.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq1/labels.npy

	echo "STARTING FCN"

	
	
	cd ../src/

	python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

	cd ../../../sbsr/fcn_model/keras_time_semantic_fcn/
	KERAS_BACKEND=tensorflow
	python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2

	mv prediction.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/fcn/seq1/prediction.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/fcn/seq1/labels.npy

	# Hanover
	cd ../../../deep_learning/LSTM-Final-Project/src/
	
	python patches_store.py -mm="ram" --debug=1 -pl 8 -po 0 -ts 0 -tnl 5000 -bs=5000 --batch_size=200 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=9 --log_dir="../data/summaries/" --path="../hn_data/" --im_h=6231 --im_w=4548 --band_n=2 --t_len=24 --id_first=1 -tof=False -nap=5000 -psv=True

	cd ../../../sbsr/fcn_model/keras_time_semantic_fcn/
	KERAS_BACKEND=tensorflow
	python main.py -pl=8 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/hn_data/" -tl=24 -cn=9 -chn=2

	mv prediction.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/${folder}/fcn/prediction.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/${folder}/fcn/labels.npy

	cd ../../../deep_learning/LSTM-Final-Project/src/
	
	python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -tof=False -nap=10000 -psv=True

	cd ../../../sbsr/fcn_model/keras_time_semantic_fcn/
	KERAS_BACKEND=tensorflow
	python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2

	mv prediction.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/fcn/seq2/prediction.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/fcn/seq2/labels.npy



fi
if [ $mode -eq 9 ]; then
	folder='normy3_avgpool'
	folder_test='normy3'
	echo "SARTING LSTM FOR CAMPO VERDE"

	cd ../batched_rnn_src/

	#python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=9 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=8 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder_test}/test_patches/seq2/"
	#mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq2/predicted.npy
	#mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq2/labels.npy

	python main.py -mm="ram" --debug=1 -pl 15 -po 0 -ts 0 -tnl 90 -bs=50000 --batch_size=128 --filters=256 -m="convlstm" --phase="train" -sc=True --class_n=10 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -psv=False -tof=False --epoch=200 -nap=2 -tmd="False" -tfld="/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder_test}/test_patches/seq1/"
	mv predicted.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq1/predicted.npy
	mv labels.npy /home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/${folder}/convlstm/seq1/labels.npy

fi
