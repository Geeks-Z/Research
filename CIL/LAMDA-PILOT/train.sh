#!/bin/bash
#python main.py --config=./exps/finetune.json
python main.py --config=./exps/simplecil.json
python main.py --config=./exps/der.json
python main.py --config=./exps/foster.json
python main.py --config=./exps/l2p.json
python main.py --config=./exps/dualprompt.json
python main.py --config=./exps/coda_prompt.json
python main.py --config=./exps/ease.json