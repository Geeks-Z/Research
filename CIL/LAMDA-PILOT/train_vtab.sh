#!/bin/bash
python main.py --config=./exps/finetune_vtab_B0.json
python main.py --config=./exps/simplecil_vtab_B0.json
python main.py --config=./exps/der_vtab_B0.json
python main.py --config=./exps/foster_vtab_B0.json
python main.py --config=./exps/l2p_vtab_B0.json
python main.py --config=./exps/dualprompt_vtab_B0.json
python main.py --config=./exps/coda_prompt_vtab_B0.json
python main.py --config=./exps/ease_vtab_B0.json