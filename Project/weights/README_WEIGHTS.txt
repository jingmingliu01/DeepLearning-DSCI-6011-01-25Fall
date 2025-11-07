# YOLACT++ Pretrained Weights

## Required Weight File

**Filename**: yolact_plus_resnet50_54_800000.pth
**Size**: 177 MB
**Source**: HuggingFace

## Download Methods

### Method 1: Automatic (via setup.sh)
```bash
bash setup.sh
# The script will automatically download from HuggingFace
```

### Method 2: wget
```bash
cd weights/
wget -O yolact_plus_resnet50_54_800000.pth \
    "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
```

### Method 3: curl
```bash
cd weights/
curl -L -o yolact_plus_resnet50_54_800000.pth \
    "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
```

### Method 4: Manual Download
1. Visit: https://huggingface.co/dbolya/yolact-plus-resnet50
2. Click "Files and versions"
3. Download: yolact_plus_resnet50_54_800000.pth
4. Place in this directory (Project/weights/)

## Verification

After downloading, verify the file:
```bash
ls -lh weights/yolact_plus_resnet50_54_800000.pth
# Should show approximately 177 MB
```

## Notes

- ❌ OLD (deprecated): Google Drive download no longer works
- ✅ NEW: All YOLACT weights are now hosted on HuggingFace
- This is the YOLACT++ ResNet50 model trained on COCO dataset
- Model performance: 34.1 mAP @ 33.5 FPS
