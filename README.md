## Research

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Research&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Research" /> <img src="https://img.shields.io/github/license/Geeks-Z/Research" /></div>

## 仓库定位

基于研究方向整理的代码仓库，包括但不限于：CIL（Class-Incremental Learning）类增量学习、PEFT（参数高效微调）等。

## 文件目录

- `📁 CIL`：CIL（Class-Incremental Learning）类增量学习【Life-Long Machine Learning/Continual Learning】
  - `📁 dualprompt-pytorch-master`：[DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | [论文源码](https://github.com/google-research/l2p) | ECCV 2022 
  - `📁 ECCV22-FOSTER-master`：[FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | [论文源码](https://github.com/G-U-N/ECCV22-FOSTER) | ECCV 2022
  - `📁 Learn-to-prompt-for-Continual-Learning-main`：[Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | [论文源码](https://github.com/google-research/l2p) | CVPR 2022
  - `📁 Libraries`：CIL Toolbox
    - `📁 PyCIL-master`：[PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) | [论文源码](https://github.com/G-U-N/PyCIL) | 
    - `📁 LAMDA-PILOT-main`：[PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://arxiv.org/abs/2309.07117) | [论文源码](https://github.com/sun-hailong/LAMDA-PILOT) | 
  - `📁 MyLibraries`：基于第三方库修改后的个人代码
---
- `📁 Model`：经典模型代码
- `📁 PEFT`：参数高效微调代码

## 实验脚本

- 执行脚本：`nohup ./train.sh > ./res/.out 2>&1 &`
- 脚本配置：编辑`train.sh`
  ```bash
  #!/bin/bash
  python main.py --config ./exps/simplecil.json
  ```