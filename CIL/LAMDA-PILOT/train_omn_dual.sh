#!/bin/bash
#python main.py --config=./exps/finetune_omn_B0_Inc30.json
#python main.py --config=./exps/simplecil_omn_B0_Inc30.json
#python main.py --config=./exps/l2p_omn_B0_Inc30.json
python main.py --config=./exps/dualprompt_omn_B0.json
python main.py --config=./exps/coda_prompt_omn_B0.json
#python main.py --config=./exps/ease_omn_B0_Inc30.json