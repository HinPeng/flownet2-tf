#!/bin/bash
#source /home/fanyang/v-xuapen/tf_1.2.1/bin/activate
rm -rf ./logs/flownet_s_sample
CUDA_VISIBLE_DEVICES=$1 python -m src.flownet_s.train --num_clones=$2
