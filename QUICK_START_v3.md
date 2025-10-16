# Quick Start Guide - Version 3.0

## 🚀 What's New in v3.0?

Three major features to supercharge your workflow:

1. **🎥 Live Recognition** - Real-time screen monitoring
2. **💾 Annotation Export** - Save detections with full metadata
3. **🖥️ GPU Acceleration** - 5-10x faster training

---

## 10-Second Quick Start

### Live Recognition
1. Go to **Recognize** tab
2. Check **"Enable Live Recognition"**
3. Watch real-time detections!

### Save with Annotations
1. Run recognition
2. Click **"💾 Save Images + Annotations"**
3. Choose format, done!

### GPU Training
1. App auto-detects GPU ✓
2. Check status bar for device
3. Train as usual (much faster!)

---

## Detailed Walkthroughs

### 🎥 Live Recognition in 5 Steps

**Goal:** Monitor screen in real-time for objects

1. **Open Recognize Tab**
   - Navigate to the Recognize tab at the top

2. **Select Your Model**
   - Choose trained model from dropdown
   - If none available, train a model first

3. **Enable Live Mode**
   - Check ✓ "Enable Live Recognition"
   - Detections start appearing immediately

4. **Adjust Settings** (Optional)
   - Move FPS slider (1-10)
   - Lower = smoother, Higher = more responsive
   - Default 3 FPS works well for most cases

5. **Save Frames** (Optional)
   - Click "Save Images + Annotations"
   - Enter number of frames (default 10)
   - Frames captured and saved automatically

**When to Use:**
- Monitoring dynamic content
- Creating training data from video
- Live demonstrations
- Real-time quality assurance

**Tips:**
- Start with 3 FPS, adjust if needed
- Lower FPS if system lags
- Live mode works with any trained model
- Status bar shows "Live: X detections"

---

### 💾 Export Detections in 3 Steps

**Goal:** Save detected images with annotations for training

1. **Run Detection**
   - Single mode: Click "Capture & Recognize"
   - Live mode: Enable live recognition
   - Verify detections appear in results list

2. **Choose Format**
   - **COCO JSON** (Recommended)
     - Standard format
     - Works with major frameworks
     - Single annotations file
   
   - **Per-Image JSON**
     - One JSON per image
     - Simple to parse
     - Good for custom processing

3. **Save**
   - Click "💾 Save Images + Annotations"
   - Check notification for location
   - Find in `exports/recognition_[timestamp]/`

**Output Structure:**
```
exports/
└── recognition_2025-10-16_09-00-00/
    ├── images/
    │   └── detection_001.png
    └── annotations/
        └── instances.json (or per-image JSONs)
```

**What's Saved:**
- Images (PNG format)
- Bounding boxes (exact coordinates)
- Class labels (detected objects)
- Confidence scores (0.0 - 1.0)
- Timestamps (when captured)
- Metadata (dimensions, source)

**Use For:**
- Training new models
- Fine-tuning existing models
- Building datasets
- Sharing detections
- Backup/archive

---

### 🖥️ GPU Training in 2 Steps

**Goal:** Train models 5-10x faster with GPU

1. **Check Device** (Automatic)
   - Look at status bar (bottom left)
   - Should show: "🖥️ Device: GPU (NVIDIA...)"
   - If shows "CPU", GPU not detected (still works!)

2. **Train Normally**
   - Go to Train tab
   - Set parameters (or use Auto-Configure)
   - Click "Start Training"
   - Training uses GPU automatically!

**Override Device** (Optional):
- Train tab → Device dropdown
- Choose: Auto / Force CPU / Force GPU
- Auto recommended for most users

**Troubleshooting:**
- **No GPU detected?**
  - Check if you have NVIDIA GPU
  - Install CUDA: nvidia.com/cuda
  - Reinstall PyTorch with CUDA support
  - Restart application

- **GPU initialization failed?**
  - App falls back to CPU automatically
  - Close other GPU applications
  - Reduce batch size
  - Try Force CPU mode

**Performance:**
- Small dataset: 5 min → 1 min (5x faster)
- Medium dataset: 20 min → 2 min (10x faster)
- Large dataset: 2 hrs → 15 min (8x faster)

---

## Common Workflows

### Workflow 1: Live Data Collection
**Scenario:** Build training dataset from dynamic content

1. Open project with classes defined
2. Go to Recognize tab
3. Train initial model (or use existing)
4. Enable "Live Recognition"
5. Set FPS to 3-5
6. Click "Save Images + Annotations"
7. Enter frames to capture (e.g., 50)
8. Review saved data in exports/
9. Move good images to project for training
10. Retrain with new data

### Workflow 2: GPU-Accelerated Training
**Scenario:** Train large model quickly

1. Create/open project
2. Label images (use Label tab)
3. Go to Train tab
4. Click "Auto-Configure Settings"
5. Verify Device shows GPU
6. Click "Start Training"
7. Watch status bar for progress
8. Training completes 5-10x faster!

### Workflow 3: Export for External Training
**Scenario:** Use detections with Detectron2/MMDetection

1. Run recognition on dataset
2. Select "COCO JSON" format
3. Click "Save Images + Annotations"
4. Find export in exports/ folder
5. Copy to training pipeline:
   ```python
   from detectron2.data.datasets import load_coco_json
   load_coco_json(
       "path/to/annotations/instances.json",
       "path/to/images/",
       "my_dataset"
   )
   ```
6. Train with standard pipeline

### Workflow 4: Live Monitoring
**Scenario:** Monitor screen for specific objects continuously

1. Train model on objects of interest
2. Go to Recognize tab
3. Select model
4. Set confidence threshold (e.g., 0.7 for high confidence)
5. Enable NMS checkbox
6. Enable "Live Recognition"
7. Adjust FPS (lower for stable monitoring)
8. Watch detections in real-time
9. Save interesting frames when needed

---

## Feature Comparison Matrix

| Feature | v2.0 | v3.0 |
|---------|------|------|
| Single Screenshot | ✓ | ✓ |
| Live Recognition | ✗ | ✓ |
| Save Images | ✓ | ✓ |
| Save Annotations | ✗ | ✓ |
| COCO JSON Export | ✗ | ✓ |
| Per-Image JSON | ✗ | ✓ |
| GPU Training | Manual | Auto |
| Device Override | ✗ | ✓ |
| Status Bar | ✗ | ✓ |
| Real-time Notifications | ✗ | ✓ |

---

## Configuration Defaults

All new features have sensible defaults:

| Setting | Default | Range | Recommendation |
|---------|---------|-------|----------------|
| Live Mode | OFF | ON/OFF | Enable when needed |
| FPS | 3 | 1-10 | 3-5 for most cases |
| Frames to Save | 10 | 1-999 | 10-50 typical |
| Export Format | COCO JSON | COCO/Per-image | COCO for training |
| Device | Auto | Auto/Force CPU/GPU | Auto recommended |

---

## Keyboard Shortcuts (Same as v2.0)

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save annotations |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `B` | Box mode |
| `P` | Polygon mode |
| `Delete` | Remove annotation |
| `←/→` | Navigate images |
| `Ctrl +/-` | Zoom |
| `F1` | Help |

---

## Tips & Tricks

### For Live Recognition
- 💡 Lower FPS for CPU-intensive models
- 💡 Use live mode to test model performance
- 💡 Save frames when interesting content appears
- 💡 Disable live mode when not needed (saves resources)

### For Annotation Export
- 💡 Use COCO JSON for training pipelines
- 💡 Use Per-Image JSON for custom processing
- 💡 Check exports/ folder regularly
- 💡 Export filename includes timestamp

### For GPU Training
- 💡 Let Auto mode handle device selection
- 💡 Check status bar to confirm GPU usage
- 💡 Increase batch size on GPU (2→4→8)
- 💡 Close other GPU apps during training

---

## Troubleshooting Quick Reference

### Problem: Live mode laggy
- **Solution**: Lower FPS slider
- **Solution**: Close other applications
- **Solution**: Use simpler model

### Problem: No detections saved
- **Solution**: Run recognition first
- **Solution**: Check if results list shows detections
- **Solution**: Click "Capture & Recognize" or enable live mode

### Problem: GPU not detected
- **Solution**: Install CUDA toolkit
- **Solution**: Reinstall PyTorch with CUDA
- **Solution**: Restart application
- **Solution**: Use CPU mode (still works!)

### Problem: Export failed
- **Solution**: Check disk space
- **Solution**: Verify write permissions
- **Solution**: Try different export location
- **Solution**: Check error message in status bar

---

## Next Steps

### After Quick Start
1. ✅ Explore live recognition with your models
2. ✅ Export detections in both formats
3. ✅ Verify GPU acceleration (if available)
4. ✅ Review FEATURES_v3.md for deep dive
5. ✅ Check CHANGELOG_v3.md for all changes

### Advanced Topics
- Multi-class detection with live mode
- Building custom training pipelines
- Optimizing GPU memory usage
- Batch processing with exports
- Integration with other tools

### Get Help
- Press **F1** in app for comprehensive help
- Read **FEATURES_v3.md** for detailed guides
- Check **README.md** for overview
- Open GitHub issue for bugs/features

---

## Feedback

Help us improve! Share your experience:
- What workflows do you use?
- What features would you like?
- What could be clearer?
- Report bugs on GitHub Issues

---

**Ready to go? Open the app and try the new features!** 🚀

*Image Labeling Studio Pro v3.0 - Professional object detection made easy.*
