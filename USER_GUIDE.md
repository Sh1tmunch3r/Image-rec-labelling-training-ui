# Image Labeling Studio Pro - User Guide 📚

## Table of Contents
1. [Getting Started](#getting-started)
2. [Annotation Workflow](#annotation-workflow)
3. [Advanced Features](#advanced-features)
4. [Training Models](#training-models)
5. [Tips & Tricks](#tips--tricks)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch
When you first launch Image Labeling Studio Pro, you'll see a welcome dialog that explains the key features and keyboard shortcuts. Take a moment to read through it!

### Creating Your First Project
1. Click the **"New Project"** button in the top bar
2. Enter a descriptive name for your project (e.g., "cats_dogs_classifier")
3. Your project will be created with folders for images and annotations

### Adding Classes
Before annotating, you need to define the classes (labels) you'll use:
1. In the **Label** tab, find the "Classes" section
2. Click **"+ Add Class"**
3. Enter the class name (e.g., "cat", "dog", "person")
4. Repeat for all your classes

## Annotation Workflow

### Loading Images
You have two options for loading images:

**Option 1: Capture Screenshot**
- Click **"📷 Capture"** button
- The app will capture your entire screen
- Useful for annotating images from other applications

**Option 2: Load from File**
- Click **"📁 Load"** button
- Select one or more image files
- Supports PNG, JPG, and JPEG formats

### Creating Bounding Box Annotations
1. Press **B** or select **"Bounding Box"** mode
2. Click and drag on the image to draw a rectangle
3. Release the mouse button
4. In the dialog, select an existing class or type a new one
5. Click **OK** to save the annotation

### Creating Polygon Annotations
1. Press **P** or select **"Polygon"** mode
2. Click multiple points around your object
3. **Right-click** when you want to close the polygon
4. Select or enter the class label
5. Click **OK** to save

### Navigating Between Images
- Use **←** and **→** arrow keys
- Or click **◀ Prev** and **Next ▶** buttons
- The counter shows your position (e.g., "5/20")

### Saving Your Work
- Press **Ctrl+S** at any time to save
- Or click **"💾 Save Annotations"**
- Annotations are saved as JSON files
- Always save before moving to the next image!

## Advanced Features

### Zoom & Pan
**Zooming:**
- **Ctrl +**: Zoom in
- **Ctrl -**: Zoom out
- **Ctrl 0**: Reset to 100%
- **Mouse wheel**: Zoom in/out
- See zoom level in the percentage indicator

**Panning:**
- **Middle-click and drag** to pan around the image
- Useful when zoomed in to see details

### Undo & Redo
- Made a mistake? Press **Ctrl+Z** to undo
- Changed your mind? Press **Ctrl+Y** to redo
- The app remembers your last 50 actions

### Copy & Paste Annotations
This is extremely useful for similar images!

1. Annotate your first image completely
2. Press **Ctrl+C** to copy all annotations
3. Navigate to the next image
4. Press **Ctrl+V** to paste the annotations
5. Adjust as needed and save

### Editing Annotations
1. Click on an annotation in the list to highlight it
2. Press **Delete** or click **"Delete"** button to remove it
3. Use undo if you delete by mistake

### Batch Operations
**For multiple similar images:**
1. Annotate a reference image
2. Copy annotations (Ctrl+C)
3. Navigate through images
4. Paste (Ctrl+V) and adjust for each
5. Save each image (Ctrl+S)

## Training Models

### Preparing for Training
Before training, ensure:
- ✓ You have at least 10-20 annotated images per class
- ✓ All annotations are saved
- ✓ Classes are properly defined

### Basic Training
1. Go to the **"Train"** tab
2. Select a preset:
   - **Fast**: Quick test (5 epochs)
   - **Balanced**: Good quality (10 epochs) - recommended
   - **Accurate**: Best results (20 epochs)
3. Click **"🚀 Start Training"**
4. Monitor the progress bar and metrics

### Advanced Training Configuration
For experienced users, customize these parameters:

**Epochs**: Number of training cycles
- More epochs = better accuracy but slower
- Start with 10-20 for most cases

**Learning Rate**: How fast the model learns
- Default: 0.005
- Lower (0.001) for fine-tuning
- Higher (0.01) for faster initial learning

**Batch Size**: Images processed together
- Default: 2
- Increase if you have GPU memory
- Decrease if you get memory errors

**Momentum**: Optimization parameter
- Default: 0.9 (usually don't change)

**Weight Decay**: Regularization
- Default: 0.0005 (usually don't change)

**Data Augmentation**: Enables random transformations
- Recommended: Keep enabled
- Helps model generalize better

### Monitoring Training
Watch the metrics console for:
- Current epoch progress
- Loss values (lower is better)
- Time per epoch
- Final model save location

### Using Your Trained Model
1. Go to the **"Recognize"** tab
2. Select your project name from the dropdown
3. Capture or load an image
4. View recognition results with confidence scores
5. Save or copy results as needed

## Tips & Tricks

### Efficiency Tips
1. **Use keyboard shortcuts** - Much faster than mouse!
2. **Copy-paste similar annotations** - Save time on similar images
3. **Zoom in for precision** - Get accurate boundaries
4. **Batch similar images** - Process related images together

### Quality Tips
1. **Be consistent** - Annotate similar objects the same way
2. **Include edges** - Don't cut off parts of objects
3. **No overlap** - Avoid overlapping bounding boxes
4. **Check your work** - Review annotations before training

### Performance Tips
1. **Save frequently** - Don't lose your work!
2. **Use presets first** - Before manual tuning
3. **More data is better** - 50+ images per class is ideal
4. **Validate annotations** - Use Dashboard > Validate Annotations

## Dashboard Features

### Statistics
The dashboard shows real-time project metrics:
- **Total Images**: All images in your project
- **Annotated**: Images with saved annotations
- **Annotations**: Total annotation count
- **Classes**: Number of unique classes

### Class Distribution
See how balanced your dataset is:
- Visual bar chart
- Count per class
- Percentage distribution
- Identify under-represented classes

### Quick Actions

**Refresh Statistics**: Update all numbers

**Validate Annotations**: Check for issues
- Missing annotations
- Invalid JSON
- Corrupt files

**Export Report**: Generate text report
- Project summary
- Class distribution
- Statistics snapshot

**Backup Project**: Create a timestamped copy
- Saves all images and annotations
- Stored in projects folder
- Safety before major changes

## Import/Export

### Exporting Annotations
1. Click **"Export"** in the top bar
2. Choose format:
   - **COCO JSON**: Popular for research
   - **YOLO TXT**: For YOLO models
   - **Pascal VOC XML**: Classic format
   - **CSV**: For spreadsheet analysis
3. Select save location
4. Annotations exported!

### Importing Annotations
1. Click **"Import"** in the top bar
2. Choose source format
3. Select file to import
4. Annotations will be converted and loaded

## Troubleshooting

### "No project selected" error
- Create a new project or open an existing one first

### Can't see my annotations
- Make sure you clicked OK after labeling
- Check if you saved (Ctrl+S)
- Verify you're looking at the correct image

### Training fails
- Check you have at least 10+ annotated images
- Verify all annotations are saved
- Make sure classes.txt is not empty
- Try reducing batch size if memory error

### Image won't load
- Check file format (PNG, JPG, JPEG only)
- Verify file isn't corrupted
- Make sure file path doesn't have special characters

### Slow performance
- Close other applications
- Reduce zoom level
- Work with fewer images at once
- Consider smaller image resolutions

### Keyboard shortcuts not working
- Make sure the main window has focus
- Click on the canvas area
- Check if another app is capturing the keys

## Best Practices

### For Best Results
1. **Consistent Quality**
   - Same level of detail across all images
   - Similar box tightness
   - Consistent class naming

2. **Balanced Dataset**
   - Similar number of images per class
   - Variety of angles and conditions
   - Include edge cases

3. **Regular Backups**
   - Backup before major changes
   - Keep multiple versions
   - Export annotations regularly

4. **Documentation**
   - Note your class definitions
   - Document special cases
   - Keep training notes

### Workflow Recommendations

**Small Projects (< 100 images)**
1. Annotate all images first
2. Review and validate
3. Train with Balanced preset
4. Evaluate results

**Large Projects (> 100 images)**
1. Annotate in batches of 20-50
2. Train intermediate models
3. Test on new images
4. Refine and continue

**Production Models**
1. Start with Fast preset for testing
2. Use Balanced for development
3. Use Accurate for final model
4. Validate on held-out test set

## Keyboard Shortcuts Reference

| Action | Shortcut | Description |
|--------|----------|-------------|
| Save | Ctrl+S | Save current annotations |
| Undo | Ctrl+Z | Undo last action |
| Redo | Ctrl+Y | Redo last undo |
| Copy | Ctrl+C | Copy annotations |
| Paste | Ctrl+V | Paste annotations |
| Delete | Delete | Remove selected annotation |
| Next Image | → | Go to next image |
| Previous Image | ← | Go to previous image |
| Zoom In | Ctrl++ | Increase zoom level |
| Zoom Out | Ctrl+- | Decrease zoom level |
| Reset Zoom | Ctrl+0 | Reset to 100% |
| Box Mode | B | Switch to box annotation |
| Polygon Mode | P | Switch to polygon annotation |

## Getting Help

Need more help?
1. Check the onboarding dialog (shown on first launch)
2. Review this user guide
3. Check README.md for technical details
4. Open an issue on GitHub
5. Read the inline tooltips in the app

---

Happy Annotating! 🎨✨
