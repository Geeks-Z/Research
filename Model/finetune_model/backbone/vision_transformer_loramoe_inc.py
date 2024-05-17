# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
from timm.models.layers import DropPath
from torch.nn import functional as F
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from collections import OrderedDict
import torch


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None, optimal_expert=None,
                 idx=None):
        super().__init__()
        self.config = config
        self.optimal_expert = optimal_expert
        # self.idx = idx
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # # 每个增量阶段初始自己的mlp
        # self.mlps = nn.ModuleList(nn.ModuleDict(
        #     {
        #         'fc1': nn.Linear(dim, mlp_hidden_dim),
        #         'fc2': nn.Linear(mlp_hidden_dim, dim),
        #     }
        # ) for i in range(config.expert_num))

    def forward(self, x, optimal_expert, idx):
        # 残差 + attention层输出
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        # mlp层输出
        # if self.mlps is not None:
        #     mlp_out = self.mlp_drop(self.act(self.mlps[optimal_expert].fc1(self.norm2(x))))
        #     mlp_out = self.drop_path(self.mlp_drop(self.mlps[optimal_expert].fc2(mlp_out)))
        #     return mlp_out + residual
        mlp_out = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        mlp_out = self.drop_path(self.mlp_drop(self.fc2(mlp_out)))
        return mlp_out + residual


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None, optimal_expert=0):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.optimal_expert = optimal_expert
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, optimal_expert=self.optimal_expert, idx=i
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, optimal_expert):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, optimal_expert, idx)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x, self.optimal_expert)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
def vit_base_patch16_224_inc(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # 控制lora参与训练的q k v 注释掉的则参与训练
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias

    # second, modify the mlp.fc.weight to match fc.weight

    # 获取预训练模型的参数（MLP）层
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            original_key = key
            fc_weight = state_dict.pop(key)
            for i in range(kwargs['tuning_config']['expert_num']):
                new_key = 'mlps.'+str(i)
                key = original_key
                state_dict[key.replace('mlp', new_key)] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # s=model.state_dict()
    # # print the keys in s
    # for key in s.keys():
    #     print(key)
    # # print the keys in checkpoint_model
    # for key in state_dict.keys():
    #     if key in s.keys():
    #         print(key, 'yes')
    #     else:
    #         print(key, 'NOOOOOOOOOOOOOOOOOOO')

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if 'fc' in name :
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model


# def vit_base_patch16_224_in21k_inc(pretrained=False, **kwargs):
#     model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#
#     # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
#     checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
#     state_dict = checkpoint_model.state_dict()
#     # modify the checkpoint state dict to match the model
#     # first, split qkv weight into q, k, v
#     # for key in list(state_dict.keys()):
#     #     if 'qkv.weight' in key:
#     #         qkv_weight = state_dict.pop(key)
#     #         q_weight = qkv_weight[:768]
#     #         k_weight = qkv_weight[768:768 * 2]
#     #         v_weight = qkv_weight[768 * 2:]
#     #         state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
#     #         state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
#     #         state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
#     #     elif 'qkv.bias' in key:
#     #         qkv_bias = state_dict.pop(key)
#     #         q_bias = qkv_bias[:768]
#     #         k_bias = qkv_bias[768:768 * 2]
#     #         v_bias = qkv_bias[768 * 2:]
#     #         state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
#     #         state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
#     #         state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
#
#     # 获取预训练模型的参数（MLP）层
#     for key in list(state_dict.keys()):
#         if 'mlp.fc' in key:
#             original_key = key
#             mlp_fc_weight = state_dict.pop(key)
#             for i in range(kwargs['tuning_config']['expert_num']):
#                 new_key = 'mlps.' + str(i)
#                 key = original_key
#                 state_dict[key.replace('mlp', new_key)] = mlp_fc_weight
#
#     # second, modify the mlp.fc.weight to match fc.weight
#     # for key in list(state_dict.keys()):
#     #     if 'mlp.fc' in key:
#     #         fc_weight = state_dict.pop(key)
#     #         state_dict[key.replace('mlp.', '')] = fc_weight
#
#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)
#
#     # freeze all
#     for name, p in model.named_parameters():
#         p.requires_grad = False
#     # freeze all but the mlp
#     # for name, p in model.named_parameters():
#     #     if 'fc' in name:
#     #         p.requires_grad = True
#     #     else:
#     #         p.requires_grad = False
#     return model
#     # s=model.state_dict()
#     # # print the keys in s
#     # for key in s.keys():
#     #     print(key)
#     # # print the keys in checkpoint_model
#     # for key in state_dict.keys():
#     #     if key in s.keys():
#     #         print(key, 'yes')
#     #     else:
#     #         print(key, 'NOOOOOOOOOOOOOOOOOOO')
#
#     # freeze all but the adapter
#     # for name, p in model.named_parameters():
#     #     if name in msg.missing_keys:
#     #         p.requires_grad = True
#     #         if 'down' in name and 'weight' in name:
#     #             nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#     #         else:
#     #             nn.init.zeros_(p)
#     #         # print(name)
#     #     else:
#     #         p.requires_grad = False
#

def vit_base_patch16_224_in21k_inc(pretrained=False, **kwargs):
    model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    for name, p in model.named_parameters():
            if 'mlp' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    return model