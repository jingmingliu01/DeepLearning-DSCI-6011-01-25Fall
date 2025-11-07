# 🎯 YOLACT++ Campus Objects Detection

**Real-time Instance Segmentation for Campus Objects using Transfer Learning**

Deep Learning Project - DSCI 6011

---

## 📖 Project Overview

This project implements a **real-time instance segmentation system** for detecting campus-specific objects using **YOLACT++ (You Only Look At CoefficienTs)**. The system is trained to identify three types of campus objects:

- 🖊️ **Whiteboard**
- 🚰 **Drinking Water Fountain**
- 🏫 **University Logo/Signage**

### Key Features

- ✅ Real-time performance (30+ FPS)
- ✅ Transfer learning with layer freezing strategy
- ✅ Custom dataset with COCO-format annotations
- ✅ Web-based demonstration interface
- ✅ End-to-end pipeline from data collection to deployment

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7-3.9
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Installation

```bash
# 1. Clone the repository
cd Project/

# 2. Create conda environment
conda create -n yolact python=3.8
conda activate yolact

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone YOLACT++
git clone https://github.com/dbolya/yolact.git

# 5. Download pretrained weights
pip install gdown
gdown 1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX -O weights/yolact_plus_resnet50_54_800000.pth
```

---

## 📊 Workflow

### 1. Data Collection & Annotation

- Collect 200-300 campus photos
- Annotate using CVAT (https://app.cvat.ai)
- Export as COCO format

**See**: [`CVAT_Annotation_Tutorial.md`](CVAT_Annotation_Tutorial.md)

### 2. Data Preparation

```bash
# Place your images in data/raw_images/
# Place COCO annotations in data/coco_annotations/instances.json

# Run preparation script
python scripts/prepare_dataset.py
```

### 3. Model Training

```bash
# Train the model
python scripts/train.py

# Training takes 2-6 hours depending on GPU
```

### 4. Model Evaluation

```bash
# Evaluate on test set
python scripts/eval_model.py
```

### 5. Inference

```bash
# Single image
python scripts/inference.py --image path/to/image.jpg

# Batch processing
python scripts/inference.py --folder path/to/images/
```

### 6. Web Deployment

```bash
# Start web application
python web_app/app.py

# Access at http://localhost:5000
```

---

## 📂 Project Structure

```
Project/
├── config.py                    # Configuration file
├── requirements.txt             # Python dependencies
├── USAGE_GUIDE.md              # ⭐ Complete usage tutorial
├── PROJECT_STRUCTURE.md        # Detailed structure
│
├── data/                       # Dataset directory
│   ├── raw_images/             # Your photos
│   ├── coco_annotations/       # CVAT annotations
│   └── processed/              # Processed data
│
├── scripts/                    # All scripts
│   ├── prepare_dataset.py      # Data preparation
│   ├── train.py               # Training
│   ├── eval_model.py          # Evaluation
│   └── inference.py           # Inference
│
├── yolact/                     # YOLACT++ code
├── weights/                    # Model weights
├── outputs/                    # Training outputs
└── web_app/                    # Web application
```

---

## 📚 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete step-by-step tutorial (⭐ START HERE)
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure
- **[CVAT_Annotation_Tutorial.md](CVAT_Annotation_Tutorial.md)** - Annotation guide
- **[YOLACT_Project_Proposal.md](YOLACT_Project_Proposal.md)** - Original proposal

---

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Classes to detect
CLASSES = ['Whiteboard', 'DrinkingWaterFountain', 'UniversityLogo']

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# Image size
IMAGE_SIZE = 550

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
```

---

## 🎯 Transfer Learning Strategy

This project uses **layer freezing** for efficient transfer learning:

**Frozen Layers** (from COCO pre-training):
- ❄️ ResNet backbone
- ❄️ Feature Pyramid Network (FPN)
- ❄️ Prototype generation network
- ❄️ Mask coefficient prediction

**Trainable Layers**:
- 🔥 Classification layer only (3 classes)

**Benefits**:
- 50-70% faster training
- Better generalization with small dataset
- Lower risk of overfitting
- Only ~1-5% of parameters trained

---

## 📈 Expected Results

### Dataset Statistics
- **Total Images**: 200-300
- **Training**: ~200 images (70%)
- **Validation**: ~50 images (20%)
- **Test**: ~30 images (10%)

### Performance Metrics
- **mAP@50**: 60-80% (expected)
- **Inference Speed**: 30+ FPS
- **Training Time**: 2-6 hours (with GPU)

---

## 🖥️ Web Application

The web interface allows you to:
- Upload images via drag-and-drop
- Real-time object detection
- Visualize segmentation masks
- Display inference time

**Screenshot**:
```
┌─────────────────────────────────────┐
│   YOLACT++ Campus Objects Detection │
├─────────────────────────────────────┤
│  Upload Image:                      │
│  ┌─────────────────────────────┐   │
│  │     Drag & Drop Here        │   │
│  │         or Click            │   │
│  └─────────────────────────────┘   │
│  [🚀 Detect Objects]                │
├─────────────────────────────────────┤
│  Original    |    Result            │
│  ┌─────────┐ | ┌─────────┐         │
│  │  Image  │ | │ Detected│         │
│  └─────────┘ | └─────────┘         │
└─────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch size in config.py
BATCH_SIZE = 4
```

**2. Model not found**
```bash
# Check if weights exist
ls weights/
```

**3. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**See [USAGE_GUIDE.md](USAGE_GUIDE.md) for more troubleshooting tips**

---

## 📝 Citation

If you use YOLACT++ in your research, please cite:

```bibtex
@article{bolya2020yolact++,
  title={YOLACT++: Better Real-time Instance Segmentation},
  author={Bolya, Daniel and Zhou, Chong and Xiao, Fanyi and Lee, Yong Jae},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```

---

## 🤝 Contributing

This is an educational project for DSCI 6011 Deep Learning course.

**Student**: Jingming Liu
**Course**: Deep Learning DSCI-6011-01
**Instructor**: Muhammad Aminul Islam

---

## 📄 License

- YOLACT++ Code: MIT License
- This Project: Educational Use

---

## 🌟 Acknowledgments

- YOLACT++ authors for the excellent codebase
- COCO dataset for pre-trained weights
- CVAT team for the annotation tool

---

## 📞 Support

For questions or issues:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Review [Common Issues](#-troubleshooting)
3. Check YOLACT++ documentation: https://github.com/dbolya/yolact

---

**🎓 Happy Learning! Good Luck with Your Project!**
