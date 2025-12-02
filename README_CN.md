# ECEM-YOLO: 基于高效裂缝增强模块的道路缺陷检测

本仓库代码及扩展数据集与投稿至《The Visual Computer》期刊的论文 **《Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection》** 直接相关。我们提供了一套完整的道路缺陷检测开源方案，包含集成**高效裂缝增强模块（ECEM）** 的改进型YOLOv11模型、训练/测试脚本以及扩展后的RDD2022数据集，旨在提升模型在复杂场景下的检测与泛化能力。

## 1. 项目概述

### 1.1 研究背景
道路缺陷（如裂缝、坑洼）严重威胁交通安全与基础设施寿命。现有检测模型在**低对比度裂缝**和**复杂道路背景**下的检测精度有限。为此，我们提出 **ECEM-YOLO** 模型，将**高效裂缝增强模块（ECEM）** 嵌入YOLOv11架构，在最小化计算开销的前提下，显著增强对细微缺陷特征的提取与识别能力。

### 1.2 核心贡献
- **提出ECEM模块**：一种轻量级注意力增强模块，专门针对低对比度裂缝特征进行优化，有效提升模型识别能力。
- **扩展RDD2022数据集**：在公开的RDD2022数据集基础上，于原图新增“线上裂缝”标注，并补充了“破损护栏”和“破损交通标志”两类新缺陷的图片，所有标注格式与原数据集保持一致。
- **支持NPU部署**：模型已适配瑞芯微RK3576芯片，满足边缘设备上实时道路检测的需求。

## 2. 环境配置

### 2.1 硬件要求
- **GPU**：NVIDIA RTX 3090/4090（推荐用于模型训练）
- **内存**：≥ 16GB

### 2.2 软件要求
- **Python** ≥ 3.10
- **PyTorch** ≥ 2.4.1
- 其他依赖库：详见 `requirements.txt`

### 2.3 安装步骤
```bash
# 1. 克隆本仓库
git clone https://github.com/zdpf122/ECEM-YOLO-RoadDefect.git
cd ECEM-YOLO-RoadDefect

# 2. 安装Python依赖
pip install -r requirements.txt
```

## 3. 数据集使用（合规性与引用）

### 3.1 数据集来源与合规性
本项目使用的数据集是 **RDD2022（Road Damage Detection 2022）** 的扩展版本，严格遵循原始数据集的授权协议与学术规范：
- **原始数据集**：[RDD2022](https://github.com/sekilab/RoadDamageDetector)
- **原始授权协议**：知识共享署名-相同方式共享 4.0 国际许可协议（CC BY-SA 4.0）
- **原始引用**：
  ```plaintext
  @article{arya2024rdd2022,
    title={RDD2022: A multi-national image dataset for automatic road damage detection},
    author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
    journal={Geoscience Data Journal},
    volume={11},
    number={4},
    pages={846--862},
    year={2024},
    publisher={Wiley Online Library}
  }
  ```
- **扩展数据集 (RFDD)**：在RDD2022基础上，原图新增“线上裂缝”标注，并新增“破损护栏”和“破损交通标志”两类图片。
- **扩展数据集授权协议**：CC BY-SA 4.0

### 3.2 数据集下载
| 数据集类型 | 访问链接 | 说明 |
| :--- | :--- | :--- |
| **完整扩展数据集** | [Zenodo](https://doi.org/10.5281/zenodo.XXXXXXX) | 用于完整模型训练与评估 |
| **示例数据集** | [GitHub ./samples/](samples/) | 用于快速验证代码功能 |

### 3.3 数据集结构
```
data/
├── images/
│   ├── train/          # 训练集图像 (JPG格式, 建议分辨率640×640)
│   └── val/            # 验证集图像
├── labels/
│   ├── train/          # 训练集标注 (YOLO txt格式: class_id x_center y_center width height)
│   └── val/            # 验证集标注
└── test/               # 测试集 (可选，结构与train/val相同)
    ├── images/
    └── labels/
```
**缺陷类别**（在RDD2022原有类别基础上扩展）：
- `D00`：纵向裂缝
- `D10`：横向裂缝
- `D20`：龟裂
- `D40`：坑洼
- `crack_on_line`：线上裂缝 *(新增)*
- `guardrail`：破损护栏 *(新增)*
- `traffic_sign`：破损交通标志 *(新增)*

## 4. 实验复现

### 4.1 模型训练
```bash
# 使用完整扩展数据集进行训练（请确保已下载数据集并正确配置data.yaml中的路径）
python train.py

# 若仅想快速测试，可使用代码库中的示例数据集
# 首先，修改 `data.yaml` 配置文件中的数据集路径指向 `./samples/`
python train.py
```

### 4.2 模型测试
```bash
# 在测试集上评估模型性能
python test.py
```

## 5. 引用要求
若您的研究中使用了本项目的代码、扩展数据集或相关成果，**请同时引用原始RDD2022数据集和我们的论文**。

1. **原始RDD2022数据集引用** (见上文 3.1 节)

2. **本工作引用** (期刊录用后将更新DOI)：
```plaintext
@article{gao202xecemyolo,
  title={Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection},
  author={Ningbo Gao, Dupengfei Zhai, Guifang Shi, Guanya Hao, Yong Qi, Zi Yang},
  journal={The Visual Computer},
  year={202X},
  publisher={Springer},
  note={Code: https://github.com/zdpf122/ECEM-YOLO-RoadDefect; Dataset: https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## 6. 授权协议
- **代码** 遵循 [MIT](LICENSE) 协议。
- **扩展数据集 (RFDD)** 遵循知识共享署名-相同方式共享 4.0 国际许可协议（CC BY-SA 4.0）。
