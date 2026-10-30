#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file=accelerate_config.yaml \
    --num_processes=4 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29500 \
    --multi_gpu \
    ./infer_flux_color_multi_gpu.py \
    --use_att \
    --use_flowedit
