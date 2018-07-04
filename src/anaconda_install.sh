#cd ~/Downloads/
#bash Anaconda3-5.2.0-Linux-x86_64.sh 

#export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
#export PATH=~/anaconda3/bin:$PATH

#conda create -y -n epyt
#source activate epyt

conda install -y -c conda-forge scikit-image
conda install -y tensorflow-gpu
conda install -y -c conda-forge opencv
conda install -y scikit-learn
conda install -y -c conda-forge keras 


sudo nano ~/.keras/keras.json