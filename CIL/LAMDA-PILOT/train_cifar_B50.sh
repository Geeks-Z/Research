#!/bin/bash
#python main.py --config=./exps/finetune_B50.json
#python main.py --config=./exps/simplecil_B50.json
#python main.py --config=./exps/der_B50.json
python main.py --config=./exps/foster_B50.json
python main.py --config=./exps/l2p_B50.json
python main.py --config=./exps/dualprompt_B50.json
python main.py --config=./exps/coda_prompt_B50.json
python main.py --config=./exps/ease_B50.json
