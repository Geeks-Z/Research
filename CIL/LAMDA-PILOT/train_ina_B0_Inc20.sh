#!/bin/bash
python main.py --config=./exps/finetune_ina_B0_Inc20.json
python main.py --config=./exps/l2p_ina_B0_Inc20.json
python main.py --config=./exps/dualprompt_ina_B0_Inc20.json
python main.py --config=./exps/coda_prompt_ina_B0_Inc20.json
python main.py --config=./exps/simplecil_ina_B0_Inc20.json
python main.py --config=./exps/ease_ina_B0_Inc20.json