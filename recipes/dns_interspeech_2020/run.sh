#!/bin/bash
conda activate FullSubNet
torchrun --standalone \
         --nnodes=1 \
         --nproc_per_node=1 FullSubNet/recipes/dns_interspeech_2020/train.py \
         -C FullSubNet/recipes/dns_interspeech_2020/fast_fullsubnet/train_shrinkSize2.toml