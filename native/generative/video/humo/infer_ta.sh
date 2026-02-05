#!/bin/bash

torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:12345 \
    --rdzv_conf=timeout=900,join_timeout=900,read_timeout=900 \
    main.py humo/configs/inference/generate.yaml \
    generation.frames=97 \
    generation.scale_a=5.5 \
    generation.scale_t=5.0 \
    generation.mode=TA \
    generation.height=720 \
    generation.width=1280 \
    diffusion.timesteps.sampling.steps=50 \
    generation.positive_prompt=./examples/test_case.json \
    generation.output.dir=./output
