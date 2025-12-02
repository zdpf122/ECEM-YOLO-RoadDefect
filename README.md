# ECEM-YOLO: Road Defect Detection with Efficient Crack Enhancement Module

> **Important Notice**: This code repository is directly related to the paper **"Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection"** submitted to *The Visual Computer* journal.
> If you use the code, models, or datasets in your research, please be sure to cite the relevant papers.
> We provide a complete open-source solution for road defect detection, including an improved YOLOv11 model integrated with the Efficient Crack Enhancement Module (ECEM), training/testing scripts, and an extended version of the RDD2022 dataset, aiming to enhance the model's detection and generalization capabilities in complex scenarios.

## 1. Project Overview

### 1.1 Research Background
Road defects (such as cracks, potholes) pose significant threats to traffic safety and infrastructure lifespan. Existing detection models show limited accuracy for **low-contrast cracks** and defects within **complex road backgrounds**. To address this, we propose the **ECEM-YOLO** model, which integrates the **Efficient Crack Enhancement Module (ECEM)** into the YOLOv11 architecture, significantly enhancing the extraction and recognition of subtle defect features while minimizing computational overhead.

### 1.2 Key Contributions
- **Proposed the ECEM Module**: A lightweight attention enhancement module specifically optimized for low-contrast crack features, effectively boosting model recognition capability.
- **Extended RDD2022 Dataset**: Based on the public RDD2022 dataset, we added "crack on line" annotations to the original images and supplemented the dataset with images containing two new defect classes: "damaged guardrail" and "damaged traffic sign". All annotations follow the same format as the original dataset.
- **NPU Deployment Support**: The model is adapted for the Rockchip RK3576 chip, meeting the requirements for real-time road defect detection on edge devices.

## 2. Environment Configuration

### 2.1 Hardware Requirements
- **GPU**: NVIDIA RTX 3090/4090 (Recommended for model training)
- **Memory**: ≥ 16GB

### 2.2 Software Requirements
- **Python** ≥ 3.10
- **PyTorch** ≥ 2.4.1
- Other dependencies: See `requirements.txt`

### 2.3 Installation Steps
```bash
# 1. Clone this repository
git clone https://github.com/zdpf122/ECEM-YOLO-RoadDefect.git
cd ECEM-YOLO-RoadDefect

# 2. Install Python dependencies
pip install -r requirements.txt
```

## 3. Dataset Usage (Compliance and Citation)

### 3.1 Dataset Source and Compliance
The dataset used in this project is an extended version of **RDD2022 (Road Damage Detection 2022)**, strictly adhering to the original dataset's license and academic norms:
- **Original Dataset**: [RDD2022](https://github.com/sekilab/RoadDamageDetector)
- **Original License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Original Citation**:
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
- **Extended Dataset (RFDD)**: Based on RDD2022, we added "crack on line" annotations to the original images and introduced new images for "damaged guardrail" and "damaged traffic sign".
- **Extended Dataset License**: CC BY-SA 4.0

### 3.2 Dataset Download
| Dataset Type | Access Link | Description |
| :--- | :--- | :--- |
| **Full Extended Dataset** | [Zenodo](https://doi.org/10.5281/zenodo.17792244) | For complete model training and evaluation |
| **Sample Dataset** | [GitHub ./samples/](samples/) | For quick code verification |

### 3.3 Dataset Structure
```
data/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training annotations (YOLO txt format)
    ├── val/            # Validation annotations
    └── test/           # Test annotations

```
**Defect Classes** (Extended based on original RDD2022 classes):
- `D00`: Longitudinal Crack
- `D10`: Transverse Crack
- `D20`: Alligator Crack
- `D40`: Pothole
- `crack_on_line`: Crack in Pavement Marking *(New)*
- `guardrail`: Damaged Guardrail *(New)*
- `traffic_sign`: Damaged Traffic Sign *(New)*

## 4. Experiment Reproduction

### 4.1 Model Training
```bash
# Train using the full extended dataset (Ensure the dataset is downloaded and paths are correctly configured in data.yaml)
python train.py

# For a quick test, you can use the sample dataset included in the repository
# First, modify the dataset path in the `data.yaml` configuration file to point to `./samples/`
python train.py
```

### 4.2 Model Testing
```bash
# Evaluate model performance on the test set
python test.py
```

## 5. Citation Requirements
If you use the code, extended dataset, or related findings from this project in your research, **please cite both the original RDD2022 dataset and our paper**.

1. **Original RDD2022 Dataset Citation** (See Section 3.1 above)

2. **Citation for This Work** (DOI will be updated upon journal acceptance):
```plaintext
@article{gao202xecemyolo,
  title={Enhanced Road Infrastructure Defect Detection via Attention-Based Visual Inspection},
  author={Ningbo Gao, Dupengfei Zhai, Guifang Shi, Guanya Hao, Yong Qi, Zi Yang},
  journal={The Visual Computer},
  year={202X},
  publisher={Springer},
  note={Code: https://github.com/zdpf122/ECEM-YOLO-RoadDefect; Dataset: https://doi.org/10.5281/zenodo.17792244}
}
```

## 6. License
- **Code** is licensed under the [MIT](LICENSE) License.
- **Extended Dataset (RFDD)** is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
