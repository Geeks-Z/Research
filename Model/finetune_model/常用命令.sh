cd /home/team/zhaohongwei/Code/Research/Model/finetune_model &&
conda activate peft

#nohup ./train.sh > ./res/finetune-last-mlp-inr.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-mlp-inr-ina-CosineLinear.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-last-mlp-inr-ina.out 2>&1 &
#
#nohup ./train.sh > ./res/finetune-last-mlp-cifar.out 2>&1 &

#nohup ./train.sh > ./res/finetune-mlp-ina-focal_loss.out 2>&1 &

#nohup ./train.sh > ./res/finetune-mlp-ina-cb_loss.out 2>&1 &

nohup ./train.sh > ./res/finetune-last-mlp-ina-cb_loss.out 2>&1 &
