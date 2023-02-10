#!/usr/bin/env bash

set -x

python3 -u main.py \
    --dataset_file ava \
    --epochs 12 \
    --lr 2e-4 \
    --lr_drop 2 10\
    --batch_size 1 \
    --num_workers 2 \
    --coco_path ../coco \
    --ytvis_path ../ytvis \
    --num_queries 300 \
    --num_frames 32 \
    --with_box_refine \
    --rel_coord \
    --backbone csn50 \
    --output_dir csn50_ablation \
    --set_cost_class 12 \
    # --masks \
    # --pretrain_weights \

#! /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
