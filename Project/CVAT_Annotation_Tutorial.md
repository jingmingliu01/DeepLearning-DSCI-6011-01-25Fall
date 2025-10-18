# CVAT Annotation Tutorial for YOLACT++ Instance Segmentation

## Overview
This guide walks you through annotating your campus images for YOLACT++ instance segmentation using CVAT.

---

## Part 1: Setting Up CVAT

### Option A: Use Cloud Version (Recommended for Beginners)
1. Go to https://app.cvat.ai
2. Create free account
3. Skip to "Creating Your Project" below

### Option B: Run Locally with Docker
```bash
# Install Docker first: https://docs.docker.com/get-docker/

# Pull and run CVAT
docker run -d -p 8080:8080 -v ~/cvat_data:/home/django/data openvino/cvat

# Access at http://localhost:8080
# Default credentials: admin / admin
```

---

## Part 2: Creating Your Project

### Step 1: Create Project
1. Click **"Projects"** in top menu
2. Click **"+"** to create new project
3. Fill in details:
   - **Name:** "Campus Objects Instance Segmentation"
   - **Labels:** Add 7 labels (see below)

### Step 2: Add Labels
For each of your object types, add a label:

| Label Name | Color (suggested) | Type |
|------------|-------------------|------|
| fire_extinguisher | Red (#FF0000) | Polygon |
| whiteboard | White (#FFFFFF) | Polygon |
| microscope | Blue (#0000FF) | Polygon |
| university_signage | Yellow (#FFFF00) | Polygon |
| bike_rack | Green (#00FF00) | Polygon |
| exit_sign | Lime (#00FF00) | Polygon |
| water_fountain | Cyan (#00FFFF) | Polygon |

**How to add labels:**
1. In project settings, find "Labels" section
2. Click "+ Add label"
3. Enter label name
4. Choose color
5. Select "Polygon" as annotation type
6. Click "Done"
7. Repeat for all 7 labels

---

## Part 3: Creating a Task and Uploading Images

### Step 1: Create Task
1. Inside your project, click **"Tasks"**
2. Click **"+"** to create new task
3. Fill in:
   - **Name:** "Sample Images for Proposal" (or "Training Set", etc.)
   - **Project:** Select your project
   - **Labels:** Auto-populated from project

### Step 2: Upload Images
1. Scroll to **"Files"** section
2. Click **"Select files"** or drag & drop
3. Upload your 3 sample images
4. Click **"Submit"**

CVAT will process the images (may take a minute)

---

## Part 4: Annotating Images

### Step 1: Open Annotation Interface
1. Find your task in the task list
2. Click **"Job #1-1"** to start annotating

### Step 2: Annotation Interface Overview
```
Top Menu Bar:
├── Shape tools (left side)
│   ├── Rectangle - Don't use for instance segmentation
│   ├── Polygon - ⭐ USE THIS
│   ├── Polyline - Don't use
│   ├── Points - Don't use
│   └── Brush - Alternative option
├── Controls (right side)
    ├── Save work
    ├── Undo/Redo
    └── Finish job
```

---

## Part 5: Creating Polygon Annotations

### Method 1: Polygon Tool (Recommended)

#### For Regular Objects (fire extinguisher, exit sign):

**Step-by-step:**
1. **Select tool:** Click the **Polygon** icon (pentagon shape)
2. **Choose label:** Select appropriate label from dropdown (e.g., "fire_extinguisher")
3. **Start drawing:**
   - Click on object edge to place first point
   - Move cursor along object boundary
   - Click to place next point
   - Continue around entire object
4. **Complete polygon:**
   - Click on first point to close polygon
   - OR press **N** key to auto-complete
5. **Adjust points:**
   - Right-click polygon to select it
   - Drag individual points to fine-tune
   - Middle-click to add new points
   - Right-click point to delete it

**Tips:**
- Use 15-25 points for most objects
- More points = more accurate but slower
- Focus on corners and curves
- Straight edges need fewer points

**Example: Fire Extinguisher**
```
Points sequence (clockwise from top):
1. Top of cylinder
2. Right edge of top cap
3. Right edge of cylinder (3-4 points along length)
4. Right edge of handle/hose
5. Bottom right
6. Bottom left
7. Left edge of handle/hose
8. Left edge of cylinder (3-4 points)
9. Left edge of top cap
10. Back to point 1
```

#### For Complex Objects (microscope, bike rack):

Use more points (25-40) and take your time:

**Microscope Example:**
```
Areas needing attention:
- Eyepiece tube (top)
- Body/arm connection
- Objective lenses
- Stage
- Base
- Fine adjustment knobs

Break it down:
1. Outline main body first
2. Add detail at complex areas
3. Use zoom (scroll wheel) for precision
```

**Bike Rack Example:**
```
For wave-style rack:
- Follow each wave peak and valley
- Include support posts
- Include bottom rail
- May need 30-40 points for full rack
```

### Method 2: AI-Assisted Polygon (CVAT Auto-annotation)

If available in your CVAT version:
1. Click **"AI Tools"** menu
2. Select **"Mask to polygon"**
3. Click roughly inside object
4. CVAT generates initial mask
5. **Always manually refine!** AI isn't perfect

---

## Part 6: Annotation Best Practices

### Do's ✅
1. **Zoom In:** Use scroll wheel to zoom for precision
2. **Include All Parts:** Don't miss handles, brackets, shadows
3. **Be Consistent:** Same level of detail across all instances
4. **Follow True Boundary:** Stay on actual edge, not where you think it should be
5. **Save Often:** Click save icon every 5-10 annotations
6. **Label Correctly:** Double-check you selected right label

### Don'ts ❌
1. **Don't Rush:** Quality over speed
2. **Don't Skip Partial Objects:** If 50%+ visible, annotate it
3. **Don't Include Other Objects:** Only outline target object
4. **Don't Make Sharp Angles:** Unless object actually has them
5. **Don't Annotate Reflections/Shadows:** Only the actual object

### Quality Checklist for Each Annotation:
- [ ] Polygon follows object boundary accurately
- [ ] No gaps or overlaps with object edge
- [ ] Appropriate number of points (not too few, not excessive)
- [ ] Correct label assigned
- [ ] All visible parts included
- [ ] No other objects accidentally included

---

## Part 7: Working with Multiple Instances

### If Image Has Multiple Same Objects:
Example: Image with 2 fire extinguishers

1. **First instance:**
   - Select "fire_extinguisher" label
   - Draw polygon around first fire extinguisher
   - Complete polygon

2. **Second instance:**
   - **Important:** Select "fire_extinguisher" label again
   - Draw polygon around second fire extinguisher
   - Complete polygon

Each instance gets its own annotation, even if same class!

### Object Hierarchy:
If objects overlap:
1. Annotate background object first
2. Then annotate foreground object
3. CVAT handles overlap automatically

---

## Part 8: Keyboard Shortcuts

Speed up annotation with these shortcuts:

| Key | Action |
|-----|--------|
| **N** | Complete current polygon |
| **Ctrl + Z** | Undo |
| **Ctrl + Shift + Z** | Redo |
| **F** | Fit image to screen |
| **Ctrl + S** | Save work |
| **+** / **-** | Zoom in/out |
| **Space** | Switch to drag mode (pan image) |
| **Esc** | Cancel current annotation |
| **]** / **[** | Next/Previous image |
| **Backspace** | Delete last point while drawing |

---

## Part 9: Review and Quality Check

### After Completing All 3 Sample Images:

1. **Go Through Each Image:**
   - Press **]** to go to next image
   - Visually inspect each annotation
   - Look for:
     - Missed objects
     - Poor boundary fitting
     - Wrong labels
     - Incomplete polygons

2. **Edit if Needed:**
   - Right-click annotation to select
   - Choose "Edit" mode
   - Adjust points
   - Save changes

3. **Statistics Check:**
   Click **"Menu"** → **"Info"** to see:
   - Total annotations: Should match number of objects
   - Per-label counts: Verify all object types present

---

## Part 10: Exporting Annotations

### Step 1: Open Task
1. Go back to task list
2. Find your task

### Step 2: Export
1. Click three dots **"⋮"** next to task
2. Select **"Export task dataset"**

### Step 3: Choose Format
**Critical:** Select **"COCO 1.0"** format
- This is the format YOLACT++ requires
- Don't use COCO 1.1 or other variants

### Step 4: Download
1. Click **"Export"**
2. Wait for processing
3. Download ZIP file
4. Extract to find `instances_default.json`

---

## Part 11: Verify Export

### Check JSON Structure:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [
        [x1, y1, x2, y2, ..., xn, yn]  ← Polygon coordinates
      ],
      "bbox": [x, y, width, height],
      "area": 12345.67,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "fire_extinguisher",
      "supercategory": "safety"
    }
  ]
}
```

### Validation Script:
```python
import json

# Load your exported JSON
with open('instances_default.json', 'r') as f:
    data = json.load(f)

# Check structure
print(f"Number of images: {len(data['images'])}")
print(f"Number of annotations: {len(data['annotations'])}")
print(f"Number of categories: {len(data['categories'])}")

# Verify categories
print("\nCategories:")
for cat in data['categories']:
    cat_anns = [a for a in data['annotations'] if a['category_id'] == cat['id']]
    print(f"  {cat['name']}: {len(cat_anns)} instances")
```

---

## Part 12: Creating Annotation Visualizations for Proposal

### Method 1: Using CVAT Screenshots
1. In annotation interface, ensure annotations are visible
2. Use screenshot tool (Windows: Snipping Tool, Mac: Cmd+Shift+4)
3. Capture image with annotation overlays
4. Save as PNG for proposal

### Method 2: Using Python Visualization
```python
from pycocotools.coco import COCO
import cv2
import numpy as np

# Load COCO annotations
coco = COCO('instances_default.json')

# Load image
img_id = 1  # First image
img_info = coco.loadImgs(img_id)[0]
img = cv2.imread(img_info['file_name'])

# Get annotations for this image
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Draw masks
mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
for ann in anns:
    # Get segmentation polygon
    segmentation = ann['segmentation'][0]
    polygon = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
    
    # Random color for each instance
    color = np.random.randint(0, 255, 3).tolist()
    
    # Draw filled polygon
    cv2.fillPoly(mask, [polygon], color)
    
    # Draw polygon outline
    cv2.polylines(img, [polygon], True, color, 2)

# Blend image with mask
result = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

# Save visualization
cv2.imwrite('annotated_sample.png', result)
```

---

## Part 13: Common Issues and Solutions

### Issue 1: Polygon Won't Close
**Solution:** 
- Click exactly on first point
- Or press **N** to auto-complete
- Ensure you have at least 3 points

### Issue 2: Accidentally Clicked Wrong Spot
**Solution:**
- Press **Backspace** to delete last point
- Or press **Esc** to cancel entire annotation
- Start over

### Issue 3: Can't See Annotations
**Solution:**
- Check "Show annotations" toggle is ON
- Adjust opacity slider
- Try different color for label

### Issue 4: Object Partially Out of Frame
**Solution:**
- Still annotate visible portion
- Polygon should go to image edge
- CVAT handles this correctly

### Issue 5: Object Too Small to Annotate Precisely
**Solution:**
- Zoom in using scroll wheel
- Use more points for small objects
- Take your time

---

## Part 14: Time Estimates

### Per Image (with 2 objects):
- Simple objects (exit sign, signage): 5-8 minutes each
- Medium complexity (fire extinguisher): 8-12 minutes
- Complex objects (microscope, bike rack): 12-18 minutes

### For Your 3 Samples:
**Sample 1** (whiteboard + fire extinguisher):
- Whiteboard: 10 min (large but simple)
- Fire extinguisher: 10 min
- **Total: ~20 minutes**

**Sample 2** (microscope + exit sign):
- Microscope: 15 min (complex shape)
- Exit sign: 5 min (simple)
- **Total: ~20 minutes**

**Sample 3** (bike rack + university sign):
- Bike rack: 18 min (wave pattern, many points)
- University sign: 6 min
- **Total: ~24 minutes**

**Grand Total for 3 Samples: ~60-75 minutes**

---

## Part 15: Sample Annotation Description Template

For your proposal, describe each annotation like this:

```markdown
### Sample 1: [Image Name]

**Objects Annotated:**
1. **Fire Extinguisher**
   - Location: Lower right, mounted on wall
   - Polygon Points: 18
   - Challenges: Hose attachment required precise point placement
   - Annotation Time: 9 minutes

2. **Whiteboard**
   - Location: Center-left of image
   - Polygon Points: 12
   - Challenges: Including marker tray and frame edges
   - Annotation Time: 11 minutes

**COCO JSON Statistics:**
- Segmentation area - Fire Extinguisher: 55,100 pixels
- Segmentation area - Whiteboard: 382,500 pixels
- Bounding boxes: Accurate with <5% padding

**Annotation Quality Check:**
- ✅ All visible parts included
- ✅ Boundaries follow object edges precisely
- ✅ No gaps or overlaps
- ✅ Correct labels assigned
```

---

## Next Steps After Proposal Approval

1. **Collect full dataset** (250-300 images)
2. **Create systematic annotation plan:**
   - Annotate 50 images per session
   - Take breaks to maintain quality
   - Estimated total time: 40-60 hours
3. **Quality review:**
   - Review 100% of annotations
   - Fix any issues
4. **Final export and split:**
   - Train: 70%
   - Val: 20%
   - Test: 10%

---

## Additional Resources

### CVAT Documentation:
- Official Docs: https://opencv.github.io/cvat/docs/
- Video Tutorial: https://www.youtube.com/watch?v=6h7HxGL6Ct4

### COCO Format:
- Format Specification: https://cocodataset.org/#format-data
- Understanding Segmentation: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

### YOLACT++ Data Format:
- Custom Dataset Guide: https://github.com/dbolya/yolact#training-on-a-custom-dataset

---

**Remember:** Quality annotations are the foundation of a good model. Take your time and be precise! 🎯

Good luck with your annotations!
