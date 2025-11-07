#!/bin/bash
# YOLACT++ Campus Objects - Quick Setup Script
# This script helps you set up the project environment

set -e

echo "=========================================="
echo " YOLACT++ Campus Objects - Setup"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"
echo ""

# Check if conda is available
echo "🔍 Checking conda..."
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓ Conda found${NC}"
    USE_CONDA=true
else
    echo -e "${YELLOW}⚠ Conda not found, will use venv${NC}"
    USE_CONDA=false
fi
echo ""

# Create virtual environment
echo "📦 Setting up virtual environment..."
if [ "$USE_CONDA" = true ]; then
    read -p "Create conda environment 'yolact'? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda create -n yolact python=3.8 -y
        echo -e "${GREEN}✓ Conda environment created${NC}"
        echo ""
        echo -e "${YELLOW}⚠ Please activate the environment:${NC}"
        echo "   conda activate yolact"
        echo "   Then run this script again"
        exit 0
    fi
else
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
        echo ""
        echo -e "${YELLOW}⚠ Please activate the environment:${NC}"
        echo "   source venv/bin/activate"
        echo "   Then run this script again"
        exit 0
    fi
fi
echo ""

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Clone YOLACT++
echo "📥 Cloning YOLACT++ repository..."
if [ ! -d "yolact" ]; then
    git clone https://github.com/dbolya/yolact.git
    echo -e "${GREEN}✓ YOLACT++ cloned${NC}"
else
    echo -e "${YELLOW}⚠ YOLACT++ already exists, skipping${NC}"
fi
echo ""

# Create directories
echo "📁 Creating project directories..."
python3 config.py
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Download pretrained weights
echo "💾 Downloading pretrained weights..."
if [ ! -f "weights/yolact_plus_resnet50_54_800000.pth" ]; then
    echo "Installing gdown..."
    pip install gdown

    echo "Downloading weights (this may take a few minutes)..."
    cd weights/
    gdown 1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX

    # Rename if needed
    if [ -f "yolact_plus_base_54_800000.pth" ]; then
        mv yolact_plus_base_54_800000.pth yolact_plus_resnet50_54_800000.pth
    fi

    cd ..
    echo -e "${GREEN}✓ Weights downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Weights already exist, skipping${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Collect and annotate your data:"
echo "   - See CVAT_Annotation_Tutorial.md"
echo "   - Place images in: data/raw_images/"
echo "   - Place annotations in: data/coco_annotations/instances.json"
echo ""
echo "2. Prepare dataset:"
echo "   python scripts/prepare_dataset.py"
echo ""
echo "3. Train model:"
echo "   python scripts/train.py"
echo ""
echo "4. Launch web app:"
echo "   python web_app/app.py"
echo ""
echo "📚 For detailed instructions, read:"
echo "   USAGE_GUIDE.md"
echo ""
echo -e "${GREEN}🎉 Good luck with your project!${NC}"
