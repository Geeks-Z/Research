import timm
import torch

model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
checkpoint_model_state_dict = torch.load('/home/team/zhaohongwei/checkpoint/imagenetrs.pth')
model.load_state_dict(checkpoint_model_state_dict)
print(1)