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
BLUE='\033[0;34m'
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
    if ! conda env list | grep -q "^yolact "; then
        echo "Creating conda environment 'yolact'..."
        conda create -n yolact python=3.8 -y
        echo -e "${GREEN}✓ Conda environment created${NC}"
        echo ""
        echo -e "${YELLOW}⚠ Please activate the environment and run setup again:${NC}"
        echo "   conda activate yolact"
        echo "   bash setup.sh"
        exit 0
    else
        echo -e "${GREEN}✓ Conda environment 'yolact' exists${NC}"
    fi
else
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
        echo ""
        echo -e "${YELLOW}⚠ Please activate the environment and run setup again:${NC}"
        echo "   source venv/bin/activate"
        echo "   bash setup.sh"
        exit 0
    else
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    fi
fi
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠ No virtual environment activated!${NC}"
    echo "Please activate your environment first:"
    if [ "$USE_CONDA" = true ]; then
        echo "   conda activate yolact"
    else
        echo "   source venv/bin/activate"
    fi
    exit 1
fi

# Install PyTorch
echo "🔥 Installing PyTorch..."
echo -e "${BLUE}ℹ️  PyTorch installation depends on your system.${NC}"
echo ""
read -p "Do you have NVIDIA GPU with CUDA? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"
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
echo -e "${BLUE}ℹ️  Source: HuggingFace (dbolya/yolact-plus-resnet50)${NC}"
if [ ! -f "weights/yolact_plus_resnet50_54_800000.pth" ]; then
    echo "Downloading weights (177 MB, this may take a few minutes)..."

    # Try wget first
    if command -v wget &> /dev/null; then
        cd weights/
        wget -O yolact_plus_resnet50_54_800000.pth \
            "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
        cd ..
        echo -e "${GREEN}✓ Weights downloaded${NC}"
    # Try curl as fallback
    elif command -v curl &> /dev/null; then
        cd weights/
        curl -L -o yolact_plus_resnet50_54_800000.pth \
            "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
        cd ..
        echo -e "${GREEN}✓ Weights downloaded${NC}"
    else
        echo -e "${RED}❌ Neither wget nor curl is available${NC}"
        echo ""
        echo -e "${YELLOW}Please download manually:${NC}"
        echo "1. Visit: https://huggingface.co/dbolya/yolact-plus-resnet50"
        echo "2. Download: yolact_plus_resnet50_54_800000.pth"
        echo "3. Place in: weights/yolact_plus_resnet50_54_800000.pth"
    fi
else
    echo -e "${YELLOW}⚠ Weights already exist, skipping${NC}"
fi
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verified${NC}"
else
    echo -e "${RED}❌ Installation verification failed${NC}"
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
