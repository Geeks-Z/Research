#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/dap/cropdisease.yaml;
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/dap/eurosat.yaml;
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/dap/isic.yaml;
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/dap/pets.yaml;
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/dap/resisc45.yaml;
CUDA_VISIBLE_DEVICES=1 python train.py --config-file configs/dap/imagenetr.yaml