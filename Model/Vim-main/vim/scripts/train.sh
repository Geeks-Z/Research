#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python main.py \
--model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 32 \
--drop-path 0.0 \
--weight-decay 0.1 \
--num_workers 25 \
--data-path /home/team/zhaohongwei/Dataset/cifar100 \
--output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \