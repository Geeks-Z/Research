#!/bin/bash
python main.py --config=./exps/finetune_inr.json
python main.py --config=./exps/simplecil_inr.json
#python main.py --config=./exps/der_inr.json
#python main.py --config=./exps/foster_inr.json
#python main.py --config=./exps/l2p_inr_B0_Inc5.json
#python main.py --config=./exps/dualprompt_inr_B0_Inc5.json
#python main.py --config=./exps/coda_prompt_inr_B0_Inc5.json
#python main.py --config=./exps/ease_inr_B0_Inc5.json