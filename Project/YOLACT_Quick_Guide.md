# YOLACT++ Project - Quick Summary & Action Items

## ✅ FEASIBILITY CONFIRMED

Your choice of YOLACT++ for Instance Segmentation is **EXCELLENT** and meets ALL requirements!

---

## Key Information

### Model Details
- **Name:** YOLACT++ (You Only Look At CoefficienTs)
- **Task:** Instance Segmentation
- **Framework:** PyTorch ✓
- **Pre-trained Weights:** Available ✓
- **Speed:** 33.5 FPS (real-time) ✓
- **Papers:** Published in ICCV 2019 and TPAMI 2020 ✓

### What You're Doing
Fine-tuning YOLACT++ (pre-trained on COCO dataset) to detect **campus-specific objects** that are NOT in the original 80 COCO classes.

---

## Your Selected Objects (All VALID ✓)

1. **Fire Extinguisher** - NOT in COCO ✓
2. **Whiteboard/Blackboard** - NOT in COCO ✓
3. **Lab Equipment** (microscopes, etc.) - NOT in COCO ✓
4. **University Signage** - NOT in COCO ✓
5. **Bicycle Parking Stand** - NOT in COCO (only bicycle itself is in COCO) ✓
6. **Emergency Exit Sign** - NOT in COCO ✓
7. **Water Fountain** - NOT in COCO ✓

### COCO Classes to Avoid (80 total):
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## IMMEDIATE ACTION ITEMS

### 🚨 URGENT - Do These FIRST

1. **Check Shared Document** ⚠️
   - Open: Project_selection_deep_learning.docx
   - Verify NO ONE else has taken "Instance Segmentation + YOLACT++"
   - If clear, IMMEDIATELY add your entry:
     ```
     Team: Jingming Liu
     Task: Instance Segmentation
     Model: YOLACT++
     Code: https://github.com/dbolya/yolact
     Papers: 
       - YOLACT: Real-time Instance Segmentation (ICCV 2019)
       - YOLACT++: Better Real-time Instance Segmentation (TPAMI 2020)
     ```

2. **Submit to Canvas**
   - Upload proposal document
   - Add comment with your project info
   - Attach both papers (download from arXiv)

### 📸 For Proposal Completion

You need **3 SAMPLE IMAGES with ANNOTATIONS** before final submission:

**Sample 1 Requirements:**
- Photo of classroom with whiteboard and fire extinguisher
- Annotate both objects using CVAT
- Export as COCO JSON
- Take screenshot showing masks

**Sample 2 Requirements:**
- Photo of lab with microscope and exit sign
- Annotate both objects
- Export annotations
- Screenshot with masks

**Sample 3 Requirements:**
- Photo of outdoor area with bike rack and university sign
- Annotate both objects
- Export annotations
- Screenshot with masks

---

## Data Collection Strategy

### Where to Photograph:
- **Classrooms:** Whiteboards, fire extinguishers
- **Labs:** Microscopes, other equipment, exit signs
- **Hallways:** Fire extinguishers, exit signs, water fountains
- **Outdoor:** Bike racks, building signs
- **Various buildings:** University signage

### Photography Tips:
- Use high resolution (1920x1080 minimum)
- Vary angles: straight-on, slightly angled
- Different distances: close-up and far
- Various lighting: morning, afternoon, indoor/outdoor
- Include context (other objects visible)
- Ensure objects are clearly visible
- Take 3-5 photos of each object type for samples

---

## Annotation Process

### Step 1: Install CVAT
```bash
# Option A: Use cvat.ai (cloud version)
Go to: https://cvat.ai

# Option B: Run locally with Docker
docker run -d -p 8080:8080 openvino/cvat
```

### Step 2: Create Project
1. Create new project: "Campus Objects Instance Segmentation"
2. Add labels:
   - fire_extinguisher
   - whiteboard
   - microscope
   - university_signage
   - bike_rack
   - exit_sign
   - water_fountain

### Step 3: Annotate Images
1. Upload your 3 sample images
2. For each object in each image:
   - Select "Polygon" tool
   - Click around object boundary (10-20 points)
   - Complete polygon by clicking first point
   - Assign correct label
   - Fine-tune points for accuracy

### Step 4: Export
1. Export format: "COCO 1.0"
2. This creates `instances_default.json`
3. Verify structure matches YOLACT++ format

---

## Setup YOLACT++ (After Proposal Approved)

### 1. Clone Repository
```bash
git clone https://github.com/dbolya/yolact.git
cd yolact
```

### 2. Create Environment
```bash
# Using conda
conda create -n yolact python=3.8
conda activate yolact

# Install PyTorch (check pytorch.org for your CUDA version)
pip install torch torchvision

# Install dependencies
pip install cython
pip install opencv-python pillow pycocotools matplotlib
```

### 3. Download Pre-trained Weights
```bash
# Create weights directory
mkdir weights
cd weights

# Download YOLACT++ ResNet50 weights
wget https://drive.google.com/file/d/1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX/view
# (You'll need to use gdown or download manually)
```

### 4. Test Installation
```bash
# Test with pre-trained model on an image
python eval.py --trained_model=weights/yolact_plus_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --image=test_image.jpg
```

---

## GPU Requirements

### Minimum Specs:
- **Training:** 8GB VRAM (12GB recommended)
- **Inference:** 4GB VRAM

### If You Don't Have GPU:
1. **Google Colab Pro** ($10/month)
   - Provides Tesla T4 or better
   - Good for training

2. **University GPU Cluster**
   - Ask professor about access
   - Usually much more powerful

3. **AWS/Azure Free Tier**
   - Some credits available for students

---

## Timeline Breakdown

### Week 1: Proposal & Initial Setup
- ✓ Complete proposal (DONE!)
- ✓ Submit to shared doc and Canvas
- Collect 3 sample images
- Annotate samples
- Download papers

### Week 2: Full Data Collection
- Photograph 100+ images
- Cover all 7 object types
- Various conditions

### Week 3: Complete Annotation
- Annotate all images in CVAT
- Quality check all annotations
- Export to COCO format
- Split train/val/test

### Week 4-5: Model Setup
- Install YOLACT++
- Download weights
- Configure for custom dataset
- Test data loader

### Week 6-8: Training
- Fine-tune model
- Monitor metrics
- Iterate and improve

### Week 9-10: Deployment
- Create web interface
- Deploy model
- Test API

### Week 11: Presentation
- Prepare slides
- Demo video
- Practice

---

## Helpful Commands Reference

### Training
```bash
# Start training with your dataset
python train.py --config=yolact_base_config --batch_size=8

# Resume from checkpoint
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1
```

### Evaluation
```bash
# Evaluate on validation set
python eval.py --trained_model=weights/your_model.pth

# Test on single image
python eval.py --trained_model=weights/your_model.pth --image=test.jpg --score_threshold=0.3

# Test on folder
python eval.py --trained_model=weights/your_model.pth --images=test_images/:output/
```

---

## Submission Checklist

### For Proposal Submission:
- [ ] Proposal document completed ✓
- [ ] 3 sample images captured
- [ ] 3 samples annotated in CVAT
- [ ] Annotations exported as COCO JSON
- [ ] Screenshots of masks included in proposal
- [ ] Both papers downloaded and attached
  - [ ] YOLACT ICCV 2019 paper
  - [ ] YOLACT++ TPAMI 2020 paper
- [ ] Entry added to shared Google Doc FIRST
- [ ] Submitted to Canvas with comment

---

## Common Questions

**Q: Why can't I use 'bicycle'?**
A: COCO already has 'bicycle' as a class. You can use 'bike_rack' (the parking structure) because that's different.

**Q: How long will annotation take?**
A: About 10-15 minutes per image. For 3 samples with 2 objects each = ~1 hour total for proposal samples.

**Q: Can I use more than 7 object types?**
A: Yes! You can add more. Just make sure none are in COCO's 80 classes.

**Q: What if someone takes this before me?**
A: Check the shared doc immediately. If taken, choose Semantic Segmentation or Keypoint Detection instead.

**Q: Do I need to collect 300 images for the proposal?**
A: No! For the PROPOSAL, you only need 3 sample images with annotations. You'll collect the full 300 images after proposal approval.

---

## Why This Project is Great

✅ **Real-time Performance:** YOLACT++ runs at 33 fps
✅ **Well-documented Code:** Excellent GitHub repo with examples
✅ **Strong Papers:** Published in top venues (ICCV, TPAMI)
✅ **Practical Application:** Campus navigation, safety, facility management
✅ **Transfer Learning:** Pre-trained weights available
✅ **Deployment Ready:** Easy to deploy as web service
✅ **Novel Dataset:** Campus objects haven't been extensively studied

---

## Resources

### Official Links:
- **Code:** https://github.com/dbolya/yolact
- **YOLACT Paper:** https://arxiv.org/abs/1904.02689
- **YOLACT++ Paper:** https://arxiv.org/abs/1912.06218
- **CVAT Tool:** https://cvat.ai

### Tutorials:
- **YOLACT Training:** https://github.com/dbolya/yolact#training-yolact
- **Custom Dataset:** https://github.com/dbolya/yolact#training-on-a-custom-dataset
- **CVAT Tutorial:** https://opencv.github.io/cvat/docs/

---

Good luck! You've chosen an excellent project with a solid foundation. 🚀
