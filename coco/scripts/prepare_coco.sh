#!/bin/bash

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip
ln -s /path/to/annotations annotations_pytorch
ln -s /path/to/train_images train2014
ln -s /path/to/val_test_images val2014
