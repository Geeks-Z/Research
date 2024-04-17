#!/bin/bash
python main.py --config=./exps/simplecil_cub_B100.json
python main.py --config=./exps/l2p_cub_B100.json
python main.py --config=./exps/dualprompt_cub_B100.json
python main.py --config=./exps/adam_adapter_cub_B100.json
