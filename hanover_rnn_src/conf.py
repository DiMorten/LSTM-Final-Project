

conf={"band_n": 6, "t_len":6, "path": "../data/", "class_n":9}
#conf["pc_mode"]="remote"
conf["pc_mode"]="local"

conf["out_path"]=conf["path"]+"results/"
conf["in_npy_path"]=conf["path"]+"in_npy/"
conf["in_rgb_path"]=conf["path"]+"in_rgb/"
conf["in_labels_path"]=conf["path"]+"labels/"
conf["patch"]={}
conf["patch"]={"size":32, "stride":16, "out_npy_path":conf["path"]+"patches_npy/"}
conf["patch"]["overlap"]=16
conf["patch"]["ims_path"]=conf["patch"]["out_npy_path"]+"patches_all/"
conf["patch"]["labels_path"]=conf["patch"]["out_npy_path"]+"labels_all/"
conf['patch']['center_pixel']=int(np.around(conf["patch"]["size"]/2))
conf["train"]={}
conf["train"]["mask"]={}
conf["train"]["mask"]["dir"]=conf["path"]+"TrainTestMask.tif"
conf["train"]["ims_path"]=conf["path"]+"train_test/train/ims/"
conf["train"]["labels_path"]=conf["path"]+"train_test/train/labels/"
conf["test"]={}
conf["test"]["ims_path"]=conf["path"]+"train_test/test/ims/"
conf["test"]["labels_path"]=conf["path"]+"train_test/test/labels/"

conf["im_size"]=(948,1068)
conf["im_3d_size"]=conf["im_size"]+(conf["band_n"],)
if conf["pc_mode"]=="remote":
	conf["subdata"]={"flag":True,"n":3768}
else:
	conf["subdata"]={"flag":True,"n":1000}
#conf["subdata"]={"flag":True,"n":500}
#conf["subdata"]={"flag":True,"n":1000}
conf["summaries_path"]=conf["path"]+"summaries/"
conf["utils_main_mode"]=6
conf["utils_flag_store"]=False

# =============== Main configuration ================= #
print(conf)