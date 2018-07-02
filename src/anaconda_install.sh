cd Downloads/
bash Anaconda3-5.2.0-Linux-x86_64.sh 

conda create -y -n epyt
conda install -c conda-forge scikit-image
conda install tensorflow-gpu
conda install -c conda-forge opencv
conda install scikit-learn
conda install -c conda-forge keras 


sudo nano ~/.keras/keras.json