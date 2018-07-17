
mode="4"

if [ "$mode" -eq "1" ]; then
	cd ~/Downloads/
	bash Anaconda3-5.2.0-Linux-x86_64.sh 
fi
if [ "$mode" -eq "2" ]; then
	export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
	export PATH=~/anaconda3/bin:$PATH
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

	conda update cairo

	pip install -U numpy

	sudo nano ~/.keras/keras.json
fi
