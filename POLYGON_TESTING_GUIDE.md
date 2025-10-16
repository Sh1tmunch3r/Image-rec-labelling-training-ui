# Polygon Annotation Testing Guide

This document provides step-by-step instructions for testing polygon annotation functionality.

## Prerequisites

- Image Labeling Studio Pro installed and running
- A project with images loaded

## Test Case 1: Create Polygon Annotations

### Steps:
1. Open or create a project
2. Load an image in the Labeling tab
3. Press `P` or click polygon mode button to enable polygon drawing
4. Click multiple points on the image to create a polygon (at least 3 points)
5. Right-click to finish the polygon
6. Select or enter a label for the annotation
7. Verify the polygon appears on the image with the label
8. Press `Ctrl+S` or click Save to save annotations

### Expected Results:
- ✅ Polygon is drawn with visible vertices
- ✅ Polygon is saved successfully
- ✅ Status shows "✓ Saved successfully"
- ✅ Annotation appears in the results panel

## Test Case 2: Verify Annotation Counting

### Steps:
1. Create polygon annotations on 2-3 images
2. Check the statistics panel (📊 section)
3. Note the "Annotated Images" count
4. Navigate to Dashboard tab
5. Check "Annotated" card and "Total Annotations" card

### Expected Results:
- ✅ Annotated images count includes images with polygon annotations
- ✅ Total annotations includes polygon annotations
- ✅ Class distribution shows labels from polygon annotations

## Test Case 3: Dataset Validation

### Steps:
1. Create polygon annotations on several images
2. Navigate to Training tab
3. Click "✓ Validate Dataset" button
4. Review validation results

### Expected Results:
- ✅ Validation passes with polygon-only annotations
- ✅ No warnings about "Missing box data" for polygon annotations
- ✅ Status shows valid annotation count including polygons
- ✅ Classes detected include polygon annotation labels

## Test Case 4: Mixed Annotations (Boxes + Polygons)

### Steps:
1. Create box annotations on some images (press `B`, drag to draw box)
2. Create polygon annotations on other images (press `P`, click points, right-click)
3. Create both box AND polygon annotations on one image
4. Save all annotations
5. Check statistics and validate dataset

### Expected Results:
- ✅ Both box and polygon annotations are counted
- ✅ Statistics show correct total annotation count
- ✅ Validation passes for mixed annotations
- ✅ All annotations appear in results panel

## Test Case 5: COCO JSON Export

### Steps:
1. Navigate to Recognition tab (after creating some detections or use labeling annotations)
2. Save detected/labeled images
3. Choose COCO JSON format when saving
4. Locate the export folder (exports/recognition_TIMESTAMP/)
5. Open `annotations/instances.json` file
6. Search for polygon annotations

### Expected Results:
- ✅ Export completes successfully
- ✅ COCO JSON file exists
- ✅ Polygon annotations have `segmentation` field with coordinate list
- ✅ Polygon annotations have `bbox` field calculated from polygon
- ✅ `segmentation` is in format: [[x1,y1,x2,y2,x3,y3,...]]
- ✅ Box annotations don't have `segmentation` field

### Sample COCO Structure for Polygon:
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "segmentation": [[100.0, 100.0, 200.0, 100.0, 150.0, 200.0]],
  "bbox": [100, 100, 100, 100],
  "area": 10000,
  "iscrowd": 0
}
```

## Test Case 6: Per-Image JSON Export

### Steps:
1. Navigate to Recognition tab
2. Save detected images
3. Choose "Per-image JSON" format
4. Locate the export folder
5. Open an annotation JSON file for an image with polygon
6. Check the structure

### Expected Results:
- ✅ Export completes successfully
- ✅ JSON files created for each image
- ✅ Detections with polygons include `polygon` field
- ✅ `polygon` field contains array of [x,y] coordinates

### Sample Per-Image JSON:
```json
{
  "image": "image1.png",
  "width": 640,
  "height": 480,
  "detections": [
    {
      "label": "triangle",
      "polygon": [[100, 100], [200, 100], [150, 200]],
      "confidence": 0.95
    }
  ]
}
```

## Test Case 7: Training with Polygon Annotations

### Steps:
1. Create a project with polygon annotations only
2. Navigate to Training tab
3. Click "✓ Validate Dataset"
4. If dependencies are installed, try to start training
5. Monitor training logs

### Expected Results:
- ✅ Dataset validation passes
- ✅ Training starts without errors
- ✅ Images with polygon annotations are used for training
- ✅ Polygons are automatically converted to bounding boxes for training
- ✅ Training proceeds normally (if dependencies installed)

## Test Case 8: Undo/Redo with Polygons

### Steps:
1. Create a polygon annotation
2. Press `Ctrl+Z` to undo
3. Press `Ctrl+Y` to redo
4. Delete the polygon and undo

### Expected Results:
- ✅ Undo removes the polygon
- ✅ Redo restores the polygon
- ✅ Undo after delete restores the polygon

## Test Case 9: Navigation with Polygon Annotations

### Steps:
1. Create polygon annotations on image 1
2. Navigate to next image (Right arrow or Next button)
3. Create annotations on image 2
4. Navigate back to image 1 (Left arrow or Prev button)
5. Verify polygons are preserved

### Expected Results:
- ✅ Polygons are saved when navigating away (if auto-save enabled)
- ✅ Polygons are loaded when returning to image
- ✅ All polygon details (points, label) are preserved

## Test Case 10: Copy/Paste Annotations (if implemented)

### Steps:
1. Create a polygon annotation
2. Press `Ctrl+C` to copy
3. Navigate to another image
4. Press `Ctrl+V` to paste

### Expected Results:
- ✅ Polygon annotation is copied
- ✅ Pasted polygon appears on new image
- ✅ Label is preserved

## Regression Tests

Test that existing box annotation functionality still works:

### Box Annotation Tests:
1. ✅ Can create box annotations (press `B`, drag)
2. ✅ Box annotations are saved and loaded correctly
3. ✅ Box annotations appear in statistics
4. ✅ Box annotations are exported in COCO format (bbox field)
5. ✅ Box annotations are used for training
6. ✅ Can mix box and polygon annotations

## Common Issues to Watch For

1. **Polygon not finishing**: Make sure to right-click after placing points
2. **Validation warnings**: Should NOT see "Missing box data" for polygons
3. **Export missing polygons**: Check that `segmentation` field exists in COCO
4. **Training skips images**: Polygon images should be included in training
5. **Statistics not updating**: Polygon-annotated images should be counted

## Automated Tests

Run the test suite to verify:

```bash
# Test polygon export functionality
python test_polygon_export.py -v

# Test dataset validation with polygons (requires PIL)
python test_dataset_registration.py -v
```

## Notes

- Polygons require at least 3 points to be valid
- COCO format requires both `segmentation` and `bbox` for polygons
- Training with Faster R-CNN uses bounding boxes (polygons auto-converted)
- Polygon coordinates are stored in original image dimensions
- All polygon functionality is backward compatible with box annotations
