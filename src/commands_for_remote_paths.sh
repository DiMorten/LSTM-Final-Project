export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.2/lib64
export PATH=~/anaconda3/bin:$PATH

source activate py361


pip install Cython
pip install tensorflow-gpu

#cp cuda/include/cudnn.h /usr/local/cuda-9.2/include