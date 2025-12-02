ECEM-YOLO：基于高效裂缝增强模块的道路缺陷检测
本代码及扩展数据集与提交至《The Visual Computer》期刊的论文《Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection》直接相关。我们提供了一套完整的道路缺陷检测开源方案，包括集成高效裂缝增强模块（ECEM）的改进型 YOLOv11 模型、训练 / 测试脚本，以及扩展后的 RDD2022 数据集，以提升模型在复杂场景下的泛化能力。
1. 项目概述
1.1 研究背景
道路缺陷（如裂缝、坑洼）严重威胁交通安全与基础设施韧性。现有检测模型在低对比度裂缝和复杂道路背景下的检测精度有限。为此，我们提出 ECEM-YOLO 模型，将高效裂缝增强模块（ECEM） 嵌入 YOLOv11 架构，在减少计算量的前提下增强细微缺陷特征提取能力。
1.2 核心贡献
提出 ECEM 模块：轻量级注意力增强模块，专门针对低对比度裂缝优化，提升特征识别能力。
扩展 RDD2022 数据集：基于公开数据集（原始仓库：https://github.com/sekilab/RoadDamageDetector）原图新增线上裂缝标注，新增破损护栏和交通标注分类图片，标注格式与原数据集保持一致。
支持 NPU 部署：适配瑞芯微 RK3576 芯片，满足现场道路检测的实时性要求。（需要）
2. 环境配置
2.1 硬件要求
GPU：NVIDIA RTX 3090/4090（推荐用于模型训练）
内存：≥ 16GB
2.2 软件要求
Python ≥ 3.10
PyTorch ≥ 2.4.1
依赖库：详见 requirements.txt
2.3 安装步骤
bash
运行
# 克隆仓库
git clone https://github.com/zdpf122/ECEM-YOLO-RoadDefect.git
cd ECEM-YOLO-RoadDefect

# 安装依赖
pip install -r requirements.txt

3. 数据集使用（合规性与引用）
3.1 数据集来源与合规性
本项目使用的数据集是 RDD2022（Road Damage Detection 2022）的扩展版本，严格遵守原始数据集的授权协议与学术规范：
原始数据集：RDD2022（公开获取地址：https://github.com/sekilab/RoadDamageDetector）
授权协议：知识共享署名 - 相同方式共享 4.0 国际许可协议（CC BY-SA 4.0）
引用格式：Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2024). RDD2022: A multi-national image dataset for automatic road damage detection. Geoscience Data Journal, 11(4), 846-862.
扩展数据集Road Facility Defect Dataset(RFDD)：在 RDD2022 基础上，原图新增线上裂缝标注，新增破损护栏和交通标注分类图片。
授权协议：CC BY-SA 4.0
3.2 数据集下载
数据集类型	访问链接	说明
完整扩展数据集	Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX
示例数据集	GitHub: ./samples/	用于快速验证代码功能
3.3 数据集结构
plaintext
data/
├─ images/
│  ├─ train/  # JPG 格式，640×640 分辨率
│  └─ labels/  # YOLO txt 格式（class_id x_center y_center width height）
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
缺陷类别（与 RDD2022 一致）：
D00：纵向裂缝
D10：横向裂缝
D20：龟裂
D40：坑洼
crack_on_line：线上裂缝
guardrail：破损护栏
traffic_sign：破损交通标注
5. 实验复现
4.1 模型训练
bash
运行
# 使用完整扩展数据集训练
python train.py

# 使用示例数据集训练（修改data.yaml里的数据集位置）
python train.py
4.2 模型测试
bash
运行
python test.py

5. 引用要求
若你在研究中使用本代码、扩展数据集或相关结果，请同时引用原始 RDD2022 数据集与本论文：
1. 原始 RDD2022 数据集引用
plaintext
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
2. 本工作引用（期刊录用后更新 DOI）
plaintext
@article{gao202xecemyolo,
  title={Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection},
  author={Ningbo Gao, Dupengfei Zhai, Guifang Shi, Guanya Hao, Yong Qi, Zi Yang},
  journal={The Visual Computer},
  year={202X},
  publisher={Springer},
  note={Code: https://github.com/你的用户名/ECEM-YOLO-RoadDefect; Dataset: https://doi.org/10.5281/zenodo.XXXXXXX}
}
6. 授权协议
代码授权：MIT 协议（详见 LICENSE 文件）
数据集授权：知识共享署名 - 相同方式共享 4.0 国际许可协议（CC BY-SA 4.0）
