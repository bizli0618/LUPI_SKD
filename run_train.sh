#!/bin/bash

# Set visible devices to all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run 4 processes (Rank 0..3)
# Each process will take 2 GPUs (Logic inside ddp_lupi_skd.py)
torchrun --nproc_per_node=4 --master_port=29500 train/ddp_lupi_skd.py
