# Vision-Language Learning with Structured Clinical Caption Supervision for Automated Skin Lesion Classification and Melanoma Detection

## 📋 Overview

This repository contains the official implementation of the paper:

**"Vision-Language Learning with Structured Clinical Caption Supervision for Automated Skin Lesion Classification and Melanoma Detection"**

The study develops a vision-language learning framework using structured clinical caption supervision for automated skin lesion classification and melanoma detection from dermoscopic images. We compare:

- **EfficientNet-B4** (CNN baseline)
- **ViT-B/16** (Image-only baseline)
- **CLIP** (Contrastive Vision-Language Model) with structured caption supervision
- **BLIP** (Generative Vision-Language Model) with structured caption supervision

The framework reformulates diagnostic and demographic metadata as class-aware structured captions to provide controlled multimodal supervision for domain adaptation of vision-language models.

---

## 🔑 Key Features

- **Structured Caption Supervision**: Transforms clinical metadata (diagnosis, age, sex, anatomical site) into semantically consistent textual descriptions for multimodal alignment.
- **Hybrid Training Objective**: Combines supervised classification and image-text contrastive learning for domain-adaptive visual representation.
- **Unified Evaluation Protocol**: Consistent data partitions, five-fold cross-validation, and filtered cross-dataset evaluation on ISIC 2019.
- **Explainability**: Grad-CAM visualizations and structured caption generation for interpretable predictions.
- **Comprehensive Ablations**: Prompt sensitivity, loss objective, and caption type analyses.

---

## 📊 Dataset

### BCN20000
- **Source**: 18,946 dermoscopic images from Hospital Clínic de Barcelona
- **Classes**: 6 diagnostic categories (Actinic keratosis, Basal cell carcinoma, Melanoma, Melanoma metastasis, Nevus, Seborrheic keratosis)
- **Preprocessing**: 224×224 resizing, ImageNet normalization, augmentation (random crop, flip, rotation, color jitter)
- **Split**: 80% train, 10% validation, 10% test (lesion-level stratified splitting)

### ISIC 2019 (External Validation)
- **Purpose**: Cross-dataset evaluation
- **Classes**: 5 shared classes (AK, BCC, MEL, NV, BKL)
- **Samples**: 3,200 filtered images

---

## 🏗️ Model Architectures

| Model | Type | Vision Backbone | Textual Supervision |
| :--- | :--- | :--- | :--- |
| EfficientNet-B4 | CNN Baseline | EfficientNet-B4 | ❌ No |
| ViT-B/16 | ViT Baseline | ViT-B/16 | ❌ No |
| CLIP (Zero-shot) | Contrastive VLM | ViT-B/32 | ❌ No |
| CLIP (Fine-tuned) | **Contrastive VLM (Proposed)** | ViT-B/32 | ✅ Structured Captions |
| BLIP (Zero-shot) | Generative VLM | ViT-B/16 | ❌ No |
| BLIP (Fine-tuned) | **Generative VLM (Proposed)** | ViT-B/16 | ✅ Structured Captions |

---

## 📈 Results

### Main Results (6 Classes)

| Model | Accuracy | Macro-F1 | MCC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| CNN (EfficientNet-B4) | 82.97% | 0.7801 | 0.7764 | 0.9654 |
| ViT-B/16 (Image-only baseline)* | 64.62%† | 0.5221† | 0.4857† | — |
| **CLIP (Fine-tuned - Proposed)** | **85.72%** | **0.8095** | **0.8116** | **0.9738** |
| BLIP (Fine-tuned - Proposed) | 82.97% | 0.7812 | 0.7745 | 0.9639 |

† ViT-B/16 trained on 4-class subset (AK, BCC, MEL, NV) due to data availability.

### Caption Generation (BLIP)

| Metric | Zero-shot | Fine-tuned | Gain |
| :--- | :--- | :--- | :--- |
| BLEU-4 | 0.0000 | 0.6108 | +0.6108 |
| ROUGE-L | 0.0892 | 0.8329 | +0.7437 |

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.10+
# PyTorch 2.0+

#setup

# Clone the repository
git clone https://github.com/datascintist-abusufian/Structured-Caption-Supervision-for-Domain-Adaptive-Vision-Language-Learning-.git
cd Structured-Caption-Supervision-for-Domain-Adaptive-Vision-Language-Learning-

# Install dependencies
pip install -r requirements.txt


torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
timm>=0.9.0
pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.65.0
opencv-python>=4.7.0
transformers>=4.30.0

##Repository Structure

├── data/
│   ├── bcn20000/                   # Dataset (images + metadata)
│   └── isic2019/                   # External validation dataset
├── models/
│   ├── efficientnet_b4.py          # CNN baseline
│   ├── vit_b16.py                  # ViT baseline
│   ├── clip_model.py               # CLIP model
│   └── blip_model.py               # BLIP model
├── training/
│   ├── train_efficientnet.py
│   ├── train_vit.py
│   ├── train_clip.py
│   └── train_blip.py
├── evaluation/
│   ├── metrics.py                  # Classification metrics
│   ├── caption_metrics.py          # BLEU, ROUGE metrics
│   └── visualization.py            # Grad-CAM, t-SNE
├── utils/
│   ├── data_utils.py               # Data loading, preprocessing
│   ├── caption_utils.py            # Structured caption generation
│   └── split_utils.py              # Lesion-level splitting
├── configs/
│   └── config.yaml                 # Training configuration
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── training_analysis.ipynb
│   └── results_visualization.ipynb
├── results/
│   ├── figures/
│   └── logs/
├── requirements.txt
├── README.md
└── LICENSE


🔧 Usage
1. Data Preparation
python
from utils.data_utils import load_bcn20000, preprocess_data

# Load metadata and images
df, images = load_bcn20000(
    csv_path='data/bcn20000/bcn_20k_train.csv',
    image_dir='data/bcn20000/images'
)

# Preprocess and split (lesion-level)
train_df, val_df, test_df = preprocess_data(df, lesion_level=True)
2. Structured Caption Generation
python
from utils.caption_utils import generate_structured_caption

# Generate caption from metadata
caption = generate_structured_caption(
    diagnosis='melanoma',
    site='trunk',
    age=58,
    sex='male'
)
# Output: "dermoscopic image of melanoma, malignant, located on trunk, patient aged 58, male."
3. Training a Model
bash
# Train ViT-B/16 baseline
python training/train_vit.py --config configs/config.yaml --model vit_b16

# Train CLIP with structured captions
python training/train_clip.py --config configs/config.yaml --structured_captions

# Train BLIP with structured captions
python training/train_blip.py --config configs/config.yaml --structured_captions
4. Evaluation
bash
# Evaluate all models
python evaluation/evaluate_all.py --checkpoint_dir checkpoints/ --test_set test.csv
📊 Ablation Studies
Setting	Accuracy	Macro-F1
CLIP (classification-only)	84.21%	0.7924
CLIP (contrastive-only)	80.13%	0.7482
CLIP (hybrid: 0.7CE + 0.3Con)	85.72%	0.8095
BLIP (image-only)	80.15%	0.7521
BLIP (free-text captions)	81.52%	0.7684
BLIP (structured captions)	82.97%	0.7812
📝 Citation
If you use this code or findings in your research, please cite our paper:

bibtex
@article{sufian2026vision,
  title={Vision-Language Learning with Structured Clinical Caption Supervision for Automated Skin Lesion Classification and Melanoma Detection},
  author={Sufian, Md Abu and Newaz, Fairuz Lubana Binte and Niu, Mingbo},
  journal={},
  year={2026},
  volume={},
  pages={}
}
🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact
For questions, issues, or collaboration inquiries:

Corresponding Author:
Mingbo Niu
Email: ivr.niu@chd.edu.cn

First Author:
Md Abu Sufian
Email: m.sufian@uel.ac.uk

🙏 Acknowledgements
BCN20000 dataset provided by Hospital Clínic de Barcelona (CC-BY-NC)

ISIC 2019 Challenge Dataset

PyTorch, Hugging Face, and timm libraries

📖 References
bibtex
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}

@article{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  journal={arXiv preprint arXiv:2201.12086},
  year={2022}
}

@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc},
  journal={International Conference on Machine Learning},
  year={2019}
}
Repository: https://github.com/datascintist-abusufian/Structured-Caption-Supervision-for-Domain-Adaptive-Vision-Language-Learning-

text

---

## Summary

This README includes:

1. **Project Overview** – Brief description of the paper and framework
2. **Key Features** – Highlights of the proposed approach
3. **Dataset Information** – BCN20000 and ISIC 2019 details
4. **Model Table** – Summary of all models evaluated
5. **Results** – Key classification and caption generation metrics
6. **Installation** – Setup instructions
7. **Repository Structure** – Organized file tree
8. **Usage** – Code examples for data loading, training, evaluation
9. **Ablation Studies** – Summary table
10. **Citation** – BibTeX for citing the paper
11. **Contributing, License, Contact** – Standard sections

You can customize the paper title, journal, and other placeholders once the paper is officially published.
