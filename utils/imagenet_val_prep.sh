#!/bin/bash
#
# script to prepare ImageNet validation dataset
# after downloading and extracting the imagenet dataset go to the val directory and run this script
#
# 
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
#
# Prepare the validation data by moving images to subfolders:
#
wget -qO- https://raw.githubusercontent.com/adaptivetokensampling/ATS/main/utils/valdirs.sh | bash
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#
