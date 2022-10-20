import os
import tarfile
file_ext = ".tar.gz"
split = "val"
tar_files_dir = "/media/data2/Kinetics/kinetics-dataset/k600_targz/"
out_dir = "/media/data2/Kinetics/kinetics-dataset/k600/"

list_tar_files = os.listdir(tar_files_dir + split)


for tar_file_name in list_tar_files:
    if not file_ext in tar_file_name:
        print(">>>>>>>>>Unsupported file type: {}".format(tar_file_name))
    class_name = tar_file_name.split(".")[0]
    class_dir = out_dir + split + "/" + class_name
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    print("Extracting {} to {}".format(tar_file_name, class_dir))
    try:
        tar_file = tarfile.open(tar_files_dir + split + "/" + tar_file_name)
        tar_file.extractall(class_dir)
    except:
        print(">>>>>>>>>>Extracting {} failed!".format(tar_file_name, class_dir))


