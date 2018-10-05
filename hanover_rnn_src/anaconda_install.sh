
mode="4"

if [ "$mode" -eq "1" ]; then
	cd ~/Downloads/
	bash Anaconda3-5.2.0-Linux-x86_64.sh 
fi
if [ "$mode" -eq "2" ]; then
	export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
	export PATH=~/anaconda3/bin:$PATH

	#export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
fi
if [ "$mode" -eq "3" ]; then
	conda create -y -n epyt
	source activate epyt
fi
if [ "$mode" -eq "4" ]; then
	source activate epyt
	conda install -y -c conda-forge scikit-image
	conda install -y tensorflow-gpu=1.8.0
	conda install -y -c conda-forge opencv
	conda install -y scikit-learn
	conda install -y -c conda-forge keras 
	conda install -y -c conda-forge opencv
	#conda install -c anaconda scikit-image 
	conda update cairo

	pip install -y -U numpy

	sudo nano ~/.keras/keras.json
fi


export LD_LIBRARY_PATH=/home/lvc/anaconda3/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/OSLSM/code:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/python:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/code:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/python:
# (epyt) lvc@SLVC03:~/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/OSLSM/os_semantic_segmentation$ echo $PATH
# /home/lvc/anaconda3/envs/epyt/bin:/home/lvc/anaconda3/bin:/home/lvc/anaconda3/bin:/usr/local/cuda-9.2/bin:/usr/local/cuda-9.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
# (epyt) lvc@SLVC03:~/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/OSLSM/os_semantic_segmentation$ echo $LD_LIBRARY_PATH
# /home/lvc/anaconda3/lib::/usr/local/cuda-9.2/lib64:/usr/local/cuda-9.2/lib64
# (epyt) lvc@SLVC03:~/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/OSLSM/os_semantic_segmentation$ echo $PYTHONPATH
# /home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/OSLSM/code:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/python:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/OSLSM/code:/home/lvc/Documents/Jorg/deep_learning/semantic_oneshot/python:
