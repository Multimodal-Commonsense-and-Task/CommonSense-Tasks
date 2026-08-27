#!/bin/bash
#

mkdir DIODE && cd DIODE

wget http://diode-dataset.s3.amazonaws.com/train.tar.gz
wget http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz
wget https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip

tar -xvzf train.tar.gz && rm -f train.tar.gz
tar -xvzf train_normals.tar.gz && rm -f train_normals.tar.gz
unzip data_list.zip && rm -f data_list.zip