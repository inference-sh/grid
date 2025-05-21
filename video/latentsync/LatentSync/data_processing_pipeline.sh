#!/bin/bash

python -m preprocess.data_processing_pipeline \
    --total_num_workers 20 \
    --per_gpu_num_workers 10 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir temp \
    --input_dir /mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw
