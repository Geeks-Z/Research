<div align=center><img src="https://hw-universal.oss-cn-beijing.aliyuncs.com/code.gif" /></div>

<p align="center">
  <a href="#-introduction">🎉Introduction</a> •
  <a href="#-methods-reproduced">🌟Methods</a> •
  <a href="#-reproduced-results">📝Results</a> <br />
  <a href="#%EF%B8%8F-how-to-use">☄️How to Use</a> •
  <a href="#-acknowledgments">👨‍🏫Acknowledgments</a> •
  <a href="#-contact">🤗Contact</a>
</p>

---

<p align="center">
  <a href=""><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Research&left_color=green&right_color=red"></a>
  <a href=""><img src="https://img.shields.io/github/last-commit/Geeks-Z/Research"></a>
  <a href=""><img src="https://img.shields.io/github/license/Geeks-Z/Research"></a>
</p>
## 🎉 TODO

## 🌟 Methods

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240417170100.png" style="zoom: 60%;" /></div>

## 📝 Results

### 参数量的比较

> （20Epoch/Inc、batch_size=48、memory_size: 2000）

| Method      | Conference | Tunable Parameters（backbone） | All Parameters | Average Accuracy (%)(CIFAR B0 Inc5) | Training Time(s) | Test Time(s) |
| ----------- | ---------- | ------------------------------ | -------------- | ----------------------------------- | ---------------- | ------------ |
| Finetune    |            | 85860176                       |                | 38.9                                | 8162.29          | 76.17        |
| L2P         | CVPR 2022  | 122980                         |                | 85.94                               | 17062.06         | 153.57       |
| DualPrompt  | ECCV 2022  | 330340                         |                | 87.87                               | 15510.99         | 144.65       |
| CODA-Prompt | CVPR 2023  | 89715556                       |                | 89.11                               | 19431.8          | 145.41       |
| SimpleCIL   |            | 0                              |                | 87.57                               | 126.11           | 76.32        |
| Ease        | CVPR2024   | 1189632                        |                | 91.51                               | 10115.34         | 271.12       |
| Ours        |            | 414960                         |                | 97.43                               | 5065.48          | 151.13       |

### 准确率的比较

#### B0

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240423111333.png" /></div>

#### CUB B100 Inc5

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240422171344.png" style="zoom: 60%;" /></div>

#### CIFAR B100 Inc5

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240422171717.png" style="zoom: 60%;" /></div>

- #### in-r B100 Inc5

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240422171551.png" style="zoom: 60%;" /></div>

#### IN-A B100 Inc5

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240423142748.png" style="zoom: 60%;" /></div>

#### OMNI B150 Inc5

#### Memory

FNN

- 所有阶段都微调 FNN
- 每个增量阶段微调一个 FNN

### Ablation Study

- Different PTMs（可选）
  | Model name | Parameters | Accuracy |
  | --- | --- | --- |
  | ViT-B/16-IN1K| | |
  |ViT-L/16-IN1K | | |
  |ViT-B/16CLIP | | |

- LoRA_Expert 放置在不同的层

  | Position   | Tunable Parameters(backbone) | CIFAR B0 Inc5 | IN-R B0 Inc5 | CUB B0 Inc5 | VTAB B0 Inc5 | IN-A B0 Inc5 | OmniBench B0 Inc5 |
  | ---------- | ---------------------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ----------------- |
  | All        | 414960                       | 97.43         | 89.01        | 94.84       | 98.15        | 72.42        | 90.56             |
  | Odd Block  | 207480                       | 97.25         | 88.4         | 94.26       | 97.71        | 74.55        | 89.58             |
  | Even Block | 207480                       | 97.15         | 87.04        | 93.54       | 98.21        | 72.78        | 88.77             |
  | Head+Tail  | 69160                        | 96.23         | 81.32        | 94.20       | 95.56        | 63.50        | 84.52             |
  | Head       | 34580                        | 94.40         | 79.44        | 93.06       | 95.62        | 69.68        | 85.11             |
  | Tail       | 34580                        | 89.46         | 69.38        | 94.41       | 95.56        | 63.50        | 84.52             |

| Position   | Tunable Parameters(backbone) | IN-R B0 Inc10 | CUB B0 Inc10 | IN-A B0 Inc10 |
| ---------- | ---------------------------- | ------------- | ------------ | ------------- |
| All        | 829920                       |               |              |               |
| Odd Block  | 414960                       |               |              |               |
| Even Block | 414960                       |               |              |               |
| Head+Tail  | 138320                       |               |              |               |
| Head       | 69160                        |               |              |               |
| Tail       | 69160                        |               |              |               |

1. 总的类别中（尽量多一些），不同阶段的 路由准确率 和 Expert 准确率的均衡，训练时间 测试时间 不同类别数目给出一个推荐的方案

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240422195509.png" style="zoom: 60%;" /></div>

```python
# omn inc5
omn_inc5_cnn_accuracy = [93.74, 91.52, 89.81, 88.81, 88.89] # 90.554
omn_inc5_route_accuracy = [87.57, 81.42, 76.73, 74.12, 73.3] # 78.63
omn_inc5_train_time = 9056.369999999999
omn_inc5_test_time = 93.85

# omn_inc15
# 92.92
omn_inc15_cnn_accuracy = [98.75, 94.25, 94.16, 95.06, 93.59, 93.11, 93.27, 93.14, 92.79, 93.11, 91.39, 91.19, 90.82, 90.01, 89.16]
# 80.07
omn_inc15_route_accuracy = [93.0, 87.75, 87.57, 86.55, 83.98, 81.42, 80.54, 79.3, 76.73, 76.34, 74.14, 74.12, 73.28, 73.08, 73.3]
omn_inc15_train_time =  7796.07
omn_inc15_test_time = 251.85

# omn_inc20
# 92.32
omn_inc20_cnn_accuracy = [98.67, 96.5, 94.67, 95.08, 95.73, 95.33, 94.52, 94.36, 94.77, 94.66, 93.84, 92.57, 91.7, 90.38, 88.33, 87.93, 87.64, 87.36, 86.35, 85.93]
# 79.97
omn_inc20_route_accuracy = [91.33, 86.83, 87.11, 87.57, 86.72, 85.15, 84.17, 81.42, 80.85, 79.56, 78.99, 76.73, 76.35, 75.72, 73.96, 74.12, 73.33, 73.19, 73.04, 73.3]
omn_inc20_train_time =  7196.36
omn_inc20_test_time = 330.47999999999996

# omn_inc25
# 93.53
omn_inc25_cnn_accuracy = [97.08, 97.08, 97.78, 98.23, 98.25, 98.47, 98.27, 98.02, 97.91, 97.79, 97.53, 96.73, 95.76, 94.54, 93.29, 92.14, 91.4, 89.74, 88.76, 88.18, 87.35, 86.86, 86.22, 85.68, 85.28]
# 80.03
omn_inc25_route_accuracy = [89.58, 88.33, 87.64, 87.19, 87.57, 87.07, 84.98, 84.87, 84.38, 81.42, 81.14, 80.34, 79.41, 78.61, 76.73, 76.21, 75.9, 74.87, 74.23, 74.12, 73.57, 73.15, 73.09, 73.18, 73.3]
omn_inc25_train_time =  6991.99
omn_inc25_test_time = 418.23

# omn_inc30
# 87.87
omn_inc30_cnn_accuracy = [98.5, 97.0, 91.67, 90.38, 90.6, 91.99, 92.99, 93.25, 92.38, 92.29, 91.76, 91.65, 91.1, 90.27, 89.28, 88.41, 87.79, 86.47, 85.75, 85.49, 84.6, 82.91, 82.45, 82.25, 81.56, 81.49, 81.03, 80.56, 80.18, 80.12]
# 80.27
omn_inc30_route_accuracy = [87.57, 81.42, 76.73, 74.12, 73.3] # 78.63
omn_inc30_train_time = 6727.39
omn_inc30_test_time = 490.17

# omn_inc50
# 85.27
omn_inc50_cnn_accuracy = [99.17, 89.58, 93.06, 90.21, 92.17, 90.56, 91.9, 90.42, 91.48, 90.99, 91.58, 91.94, 92.11, 91.0, 91.05, 90.35, 89.35, 89.43, 89.24, 88.06, 87.99, 87.93, 87.01, 86.5, 86.14, 85.71, 85.49, 84.49, 84.22, 82.99, 82.88, 82.25, 81.85, 81.38, 81.12, 80.16, 79.0, 79.2, 79.28, 79.01, 78.3, 78.44, 78.26, 77.77, 77.67, 77.32, 76.95, 76.87, 76.88, 76.81]
# 80.47
omn_inc50_route_accuracy = [89.58, 88.33, 87.64, 87.19, 87.57, 87.07, 84.98, 84.87, 84.38, 81.42, 81.14, 80.34, 79.41, 78.61, 76.73, 76.21, 75.9, 74.87, 74.23, 74.12, 73.57, 73.15, 73.09, 73.18, 73.3] # 80.03
omn_inc50_train_time =  6663.21
omn_inc50_test_time = 828.21

# omn_inc60
# 83.15
omn_inc60_cnn_accuracy = [100.0, 97.5, 95.67, 95.25, 89.8, 86.83, 88.14, 88.0, 87.89, 87.8, 88.64, 89.32, 89.92, 90.28, 90.13, 90.24, 88.93, 89.15, 89.04, 87.63, 87.74, 86.61, 85.89, 85.39, 84.53, 84.16, 83.93, 83.47, 83.24, 82.46, 82.12, 81.93, 81.81, 81.01, 81.04, 80.07, 79.92, 79.42, 79.46, 79.32, 78.85, 78.56, 77.8, 76.99, 76.63, 76.74, 76.96, 76.73, 76.27, 76.14, 76.0, 75.82, 75.35, 75.59, 75.27, 75.3, 75.15, 75.16, 75.26, 75.25]
# 80.54
omn_inc60_route_accuracy = [100.0, 93.0, 91.33, 93.0, 88.8, 86.83, 87.86, 87.75, 87.11, 87.2, 88.0, 87.57, 88.14, 87.06, 86.72, 86.55, 84.98, 85.15, 85.19, 83.98, 84.17, 82.88, 81.93, 81.42, 81.0, 80.89, 80.85, 80.54, 80.37, 79.56, 79.37, 79.3, 78.99, 78.45, 78.15, 76.73, 76.69, 76.2, 76.35, 76.34, 75.98, 75.72, 75.03, 74.14, 73.96, 74.08, 74.38, 74.12, 73.61, 73.48, 73.33, 73.28, 72.9, 73.19, 73.03, 73.08, 73.04, 73.16, 73.26, 73.3]
omn_inc60_train_time =  6444.97
omn_inc60_test_time = 981.48
```

## ☄️ How to Use

## 👨‍🏫 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

## 🤗 Contact

\- [Geeks_Z の Blog](https://www.hwzhao.cn/)

\- [GitHub](https://github.com/Geeks-Z)
