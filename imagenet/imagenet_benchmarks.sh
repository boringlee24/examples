#!/bin/sh

python main.py -a resnet50 \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

python main.py -a vit_b_16 \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

python main.py -a vgg19_bn \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

python main.py -a resnext50_32x4d \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

python main.py -a inception_v3 \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

python main.py -a shufflenet_v2_x1_5 \
-b 256 --benchmarking --gpu-type $1 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-gpu 4 imagenet

