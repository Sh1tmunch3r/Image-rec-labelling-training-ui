# Dataset Export and Training Workflow

This document describes the improved workflow for exporting datasets from the recognition tab and using them for training.

## Overview

The application now automatically validates and registers exported datasets, making it seamless to go from recognition/capture to training without manual intervention.

## Workflow Steps

### 1. Capture and Recognize

In the **Recognize** tab:
1. Use "Capture & Recognize" to detect objects in screenshots
2. Or enable "Live Recognition" mode for continuous detection
3. Review the detected objects in the results panel
4. Adjust confidence threshold and NMS settings as needed

### 2. Export Dataset

Still in the **Recognize** tab:
1. Click "💾 Save Images + Annotations"
2. Choose export format:
   - **COCO JSON**: Industry-standard format with all metadata
   - **Per-image JSON**: Simple format, one JSON file per image
3. For live mode: specify how many frames to save
4. Click Save

### 3. Automatic Validation

After export, the application automatically:
- ✓ Validates the dataset structure (images/, annotations/ folders)
- ✓ Counts images and annotations
- ✓ Detects all class labels
- ✓ Checks for format errors or missing data
- ✓ Reports any warnings or issues

### 4. Dataset Registration (Optional but Recommended)

After validation, you'll see a dialog asking if you want to register the dataset for training:

**If you click Yes:**
- Dataset is copied to the `projects/` folder
- A `classes.txt` file is auto-generated from detected labels
- Dataset becomes immediately available for training
- You can optionally switch to this project right away

**If you click No:**
- Dataset is saved in `exports/` folder
- Can be manually registered later via Training tab

### 5. Training with Registered Dataset

In the **Train** tab:

**Option A: Use Current/Registered Project**
1. If you switched to the registered project, just click "🚀 Start Training"
2. Or open any project from the project dropdown

**Option B: Auto-Select Most Recent**
1. If no project is selected, training will auto-select the most recent project
2. You'll be asked to confirm before training starts

**Option C: Validate Before Training**
1. Click "✓ Validate Dataset" to check if current/recent project is ready
2. View detailed validation results:
   - Image count
   - Annotation count
   - Detected classes
   - Any errors or warnings
3. Then click "🚀 Start Training" if validation passes

### 6. Model Recognition

After training completes:
- The newly trained model is automatically selected in the Recognize tab
- Model appears in the recognizer dropdown with your project name
- Ready to use immediately for recognition

## Dataset Validation Details

The validator checks for:

### Critical Errors (prevents training)
- ❌ Missing `images/` or `annotations/` directories
- ❌ No images found in dataset
- ❌ No valid annotations found
- ❌ No class labels detected
- ❌ Invalid JSON format in annotation files

### Warnings (allows training but flagged)
- ⚠️ Images without annotations
- ⚠️ Empty annotation arrays
- ⚠️ Missing bounding box data in annotations

## Annotation Format Support

The application supports **both** formats interchangeably:

### Export Format (from Recognize tab)
```json
{
  "image": "frame_0001.png",
  "width": 1920,
  "height": 1080,
  "timestamp": "2025-10-16_14-30-00",
  "detections": [
    {
      "label": "object1",
      "box": [100, 100, 200, 200],
      "confidence": 0.95
    }
  ]
}
```

### Project Format (from Label tab)
```json
{
  "image_width": 1920,
  "image_height": 1080,
  "annotations": [
    {
      "label": "object1",
      "box": [100, 100, 200, 200]
    }
  ]
}
```

Both formats work seamlessly for training - the validator and dataset loader handle both automatically.

## Troubleshooting

### "No valid annotated images" Error

**Cause:** Dataset validation failed.

**Solutions:**
1. Click "✓ Validate Dataset" to see specific errors
2. Check that:
   - Images are in `images/` folder with .png, .jpg, or .jpeg extensions
   - Annotations are in `annotations/` folder with matching .json filenames
   - Each JSON file contains either `detections` or `annotations` key
   - Each detection/annotation has a `label` and `box` field

### Dataset Not Showing Up for Training

**Cause:** Dataset was exported but not registered as a project.

**Solutions:**
1. Export again and choose "Yes" when asked to register
2. Or manually copy dataset from `exports/` to `projects/` folder
3. Ensure `classes.txt` file exists in the project folder

### Training Starts with Wrong Dataset

**Cause:** Auto-selection picked the most recently modified project.

**Solutions:**
1. Explicitly select the desired project before training
2. Or validate dataset first to confirm which one will be used

### Validation Passes but Training Still Fails

**Cause:** Class mismatch or annotation format issue.

**Possible Fixes:**
1. Check `classes.txt` matches labels in annotations
2. Verify bounding boxes are in [x1, y1, x2, y2] format (not width/height)
3. Ensure at least 2-3 annotated images per class for meaningful training

## Best Practices

1. **Always Validate First**: Use "✓ Validate Dataset" before training to catch issues early
2. **Register Exports**: Register datasets immediately after export for better organization
3. **Descriptive Names**: Use meaningful project names when registering
4. **Multiple Annotations**: Capture/annotate multiple images per class for better models
5. **Review Classes**: Check `classes.txt` after registration to ensure correct labels
6. **Backup Projects**: Use Dashboard > Backup Project to save important datasets

## Advanced: Manual Registration

If you have an existing dataset in the correct format:

1. Create folder structure:
   ```
   my_dataset/
   ├── images/
   │   ├── img001.png
   │   └── img002.png
   └── annotations/
       ├── img001.json
       └── img002.json
   ```

2. Copy to projects folder:
   ```
   cp -r my_dataset projects/my_project_name
   ```

3. Create `classes.txt`:
   ```
   echo "class1" > projects/my_project_name/classes.txt
   echo "class2" >> projects/my_project_name/classes.txt
   ```

4. Open project in app and validate before training

## See Also

- [README.md](README.md) - Main documentation
- [USER_GUIDE.md](USER_GUIDE.md) - Complete user guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing documentation
