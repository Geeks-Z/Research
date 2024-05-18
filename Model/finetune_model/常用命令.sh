cd /home/team/zhaohongwei/Code/Research/Model/finetune_model

nohup ./train.sh > ./res/finetune-last-mlp-inr.out 2>&1 &

nohup ./train.sh > ./res/finetune-mlp-inr-ina-CosineLinear.out 2>&1 &

nohup ./train.sh > ./res/finetune-last-mlp-inr-ina.out 2>&1 &