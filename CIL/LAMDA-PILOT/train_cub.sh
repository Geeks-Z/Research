#!/bin/bash
python main.py --config=./exps/finetune_cub_B0.json
python main.py --config=./exps/simplecil_cub_B0.json
python main.py --config=./exps/der_cub_B0.json
python main.py --config=./exps/foster_cub_B0.json
python main.py --config=./exps/l2p_cub_B0.json
python main.py --config=./exps/dualprompt_cub_B0.json
python main.py --config=./exps/coda_prompt_cub_B0.json
python main.py --config=./exps/ease_cub_B0.json