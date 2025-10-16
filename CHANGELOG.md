# Changelog

All notable changes to Image Labeling Studio Pro are documented in this file.

## [Unreleased] - Polygon Annotation Fixes

### 🔧 Fixed Issues with Polygon Annotations

#### Annotation Validation & Counting
- **Fixed**: Polygon annotations now properly recognized as valid annotations in dataset validation
- **Fixed**: Images with polygon annotations are now correctly counted in annotation statistics
- **Fixed**: Dataset validation no longer warns about missing box data for polygon annotations
- **Enhanced**: Validation now checks for either `box` or `polygon` data in annotations

#### Export Functionality
- **Fixed**: COCO JSON export now includes polygon annotations in proper `segmentation` format
- **Fixed**: COCO export calculates bounding boxes from polygon points for COCO compatibility
- **Fixed**: Per-image JSON export now includes polygon data in detections
- **Enhanced**: Export functions handle mixed annotations (both boxes and polygons) correctly

#### Training Support
- **Fixed**: Training dataset loader now converts polygon annotations to bounding boxes for model training
- **Fixed**: Images with only polygon annotations are now included in training datasets
- **Enhanced**: Polygon annotations are automatically converted to bounding boxes for Faster R-CNN training

### 🧪 Testing
- **NEW**: `test_polygon_export.py` - Tests for polygon export in COCO and per-image JSON formats
- **Enhanced**: `test_dataset_registration.py` - Added tests for polygon annotation validation
- Tests cover: polygon validation, mixed box/polygon datasets, COCO segmentation format

### 📊 Technical Details
- Polygon annotations stored as list of [x, y] coordinate pairs
- COCO export: polygons in `segmentation` field as flattened coordinate list
- Bounding box auto-calculated from polygon for training: [min_x, min_y, max_x, max_y]
- Full backwards compatibility with existing box annotations

## [Unreleased] - Dataset Registration & Validation

### 🔧 Dataset Management (NEW)

#### Automatic Dataset Registration
- **Export-to-Training Flow**: Exported datasets from Recognize tab can now be immediately registered as training projects
- **Dataset Validation**: Comprehensive validation checks dataset structure, annotations, and classes before training
- **Auto-Registration Dialog**: After export, users can choose to register dataset as a project in one click
- **Project Switching**: Option to immediately switch to newly registered project after export
- **Format Support**: Supports both export format (`detections` key) and project format (`annotations` key)

#### Training Improvements
- **Auto-Select Dataset**: Training now auto-selects most recent project if none is selected
- **Validation Before Training**: Added "✓ Validate Dataset" button in Training tab
- **Dataset Status Display**: Shows validation status (images, annotations, classes) in Training tab
- **Better Error Messages**: Detailed diagnostics when dataset validation fails
- **Auto-Select Recognizer**: Newly trained models are automatically selected in Recognize tab

#### Validation Features
- **Comprehensive Checks**: Validates directory structure, image count, annotation count, JSON format
- **Detailed Diagnostics**: Shows specific errors (missing files, invalid JSON, format issues)
- **Warning System**: Flags non-critical issues (missing annotations, empty arrays) without blocking training
- **Class Detection**: Automatically detects and lists all classes found in annotations
- **Format Flexibility**: Accepts both `detections` and `annotations` keys in JSON files

### 📚 Documentation
- **NEW**: `DATASET_WORKFLOW.md` - Complete guide for export-to-training workflow
- **NEW**: `dataset_utils.py` - Standalone validation/registration utilities (no GUI dependencies)
- **Updated**: README with new workflow section
- **Updated**: Test suite with 13 new tests covering dataset validation and registration

### 🧪 Testing
- **NEW**: `test_dataset_registration.py` - Comprehensive test suite for:
  - Dataset validation (valid, empty, missing annotations, invalid JSON)
  - Dataset registration as project
  - Format compatibility (detections vs annotations)
  - Complete export-to-training flow
- All tests pass (13/13 new tests, 32/32 existing tests)

### 🔄 Backwards Compatibility
- Existing projects and workflows continue to work unchanged
- Old annotation format (`annotations` key) fully supported
- No breaking changes to existing functionality

## [2.0.0] - Enhanced Professional Version

### 🎨 Major UI/UX Overhaul

#### New Professional Interface
- **Renamed Application**: "Image Recognizer Pro" → "Image Labeling Studio Pro"
- **Larger Default Window**: 1100x700 → 1400x900 for better workspace
- **Enhanced Project Bar**: Now includes statistics display and quick actions
- **Modern Color Scheme**: Professional dark theme with color-coded elements
- **Emoji-Enhanced Buttons**: Visual icons for better user experience
- **Responsive Panels**: Left panel expanded from 230px → 280px for better usability

### 🎯 Annotation Features

#### Multiple Annotation Types
- **Bounding Boxes**: Original drag-and-drop functionality enhanced
- **Polygon Annotations**: NEW - Click to add points, right-click to close
- **Mode Selector**: Radio buttons to switch between Box (B) and Polygon (P) modes
- **Visual Feedback**: White dashed preview while drawing
- **Better Precision**: Zoom in for pixel-perfect annotations

#### Advanced Annotation Controls
- **Undo/Redo System**: NEW - 50-step history with Ctrl+Z/Ctrl+Y
- **Copy/Paste**: NEW - Copy annotations between images with Ctrl+C/Ctrl+V
- **Batch Operations**: Copy annotations from one image to similar images
- **Enhanced Delete**: Now works with undo system
- **Improved Label Dialog**: Better UI for selecting or creating labels

### 🔍 Image Viewing & Navigation

#### Zoom & Pan Controls
- **Zoom In/Out**: NEW - Ctrl +/- or mouse wheel
- **Pan Mode**: NEW - Middle-click and drag
- **Reset Zoom**: NEW - Ctrl+0 to return to 100%
- **Zoom Indicator**: Live percentage display
- **Zoom Range**: 10% to 500% for extreme detail work

#### Image Navigation
- **Previous/Next Buttons**: NEW - Navigate through image list
- **Arrow Key Navigation**: NEW - Use ← and → keys
- **Image Counter**: NEW - Shows current position (e.g., "5/20")
- **Auto-Load Annotations**: Automatically loads saved annotations
- **Image List Management**: Sorted, indexed image browsing

### 📊 Project Management

#### Dashboard Tab (NEW)
- **Statistics Cards**: Real-time metrics display
  - Total images count
  - Annotated images count
  - Total annotations
  - Number of classes
- **Class Distribution**: Visual bar chart with percentages
- **Quick Actions**:
  - Refresh statistics
  - Validate annotations
  - Export report
  - Backup project
- **Welcome Guide**: Interactive help text

#### Project Statistics Bar
- **Live Stats**: Updates automatically as you work
- **Compact Display**: Shows key metrics in top bar
- **At-a-Glance Info**: Images, annotations, and class counts

### 💾 Import/Export

#### Export Formats (NEW UI)
- **COCO JSON**: Research-standard format
- **YOLO TXT**: For YOLO model training
- **Pascal VOC XML**: Classic annotation format
- **CSV**: Spreadsheet-compatible export
- **Format Selection Dialog**: User-friendly format picker

#### Import Formats (NEW UI)
- **COCO JSON**: Import from research datasets
- **YOLO TXT**: Import YOLO annotations
- **Pascal VOC XML**: Import classic annotations
- **Conversion Pipeline**: Automatic format conversion

#### Project Backup (NEW)
- **One-Click Backup**: Create timestamped project copy
- **Full Project Copy**: Images + annotations + classes
- **Safety Feature**: Backup before major changes

### 🚀 Training Enhancements

#### Advanced Training UI
- **Two-Column Layout**: Settings on left, progress on right
- **Hyperparameter Controls**: All parameters now configurable
  - Epochs
  - Learning rate
  - Batch size
  - Momentum
  - Weight decay
- **Data Augmentation Toggle**: Enable/disable augmentation
- **Visual Training Button**: Large, prominent "🚀 Start Training"

#### Training Presets (NEW)
- **Fast Preset**: 5 epochs, LR 0.01, batch size 4
- **Balanced Preset**: 10 epochs, LR 0.005, batch size 2 (default)
- **Accurate Preset**: 20 epochs, LR 0.001, batch size 2
- **Auto-Apply**: Clicking preset updates all parameters

#### Real-Time Progress Visualization (NEW)
- **Progress Bar**: Visual training progress
- **Percentage Display**: Numeric completion indicator
- **Epoch Counter**: Current epoch of total
- **Loss Display**: Real-time loss values
- **Training Console**: Scrollable log of all metrics
- **Colored Status**: Blue (training), Orange (in progress), Green (complete), Red (error)

### ⌨️ Keyboard Shortcuts

#### New Shortcuts
- **Ctrl+S**: Save annotations
- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo last action
- **Ctrl+C**: Copy annotations
- **Ctrl+V**: Paste annotations
- **Delete**: Remove selected annotation
- **←/→**: Navigate between images
- **Ctrl +**: Zoom in
- **Ctrl -**: Zoom out
- **Ctrl+0**: Reset zoom
- **B**: Switch to box mode
- **P**: Switch to polygon mode

#### Shortcut System
- **Global Bindings**: Work throughout the application
- **Consistent Actions**: Standard keyboard conventions
- **Visual Feedback**: Status updates for all actions

### ♿ Accessibility & UX

#### Onboarding Experience (NEW)
- **Welcome Dialog**: Shows on first launch
- **Feature Overview**: Comprehensive feature list
- **Quick Start Guide**: Step-by-step instructions
- **Keyboard Shortcuts Reference**: Full shortcut list
- **Skip Option**: Won't show again after first view

#### User Experience Improvements
- **Status Indicators**: Real-time feedback for all actions
- **Color-Coded Feedback**:
  - Green: Success
  - Blue: Information
  - Orange: Warning
  - Red: Error
- **Emoji Icons**: Visual enhancement for buttons
- **Tooltips**: Contextual help (foundation for future enhancement)
- **Confirmation Dialogs**: Prevent accidental actions

#### Professional Polish
- **Consistent Styling**: Unified design language
- **Smooth Interactions**: No jarring UI changes
- **Clear Visual Hierarchy**: Important actions stand out
- **Responsive Layout**: Adapts to window size
- **Dark Theme**: Easy on the eyes for long sessions

### 🔧 Technical Improvements

#### Performance Optimizations
- **Lazy Loading**: Images loaded on demand
- **Efficient Rendering**: Optimized canvas operations
- **Background Training**: Non-blocking model training
- **State Management**: Efficient undo/redo with deep copy

#### Code Quality
- **Type Hints**: Added for better code documentation
- **Error Handling**: Try-catch blocks throughout
- **Validation**: Input validation for all operations
- **Logging**: Training metrics logged to console
- **Documentation**: Inline comments and docstrings

#### Data Management
- **JSON Indentation**: Pretty-printed annotation files
- **Sorted Image Lists**: Consistent ordering
- **State Preservation**: Zoom and pan state maintained
- **Auto-Update Stats**: Statistics refresh automatically

### 📚 Documentation

#### New Documentation Files
- **README.md**: Comprehensive feature overview
- **USER_GUIDE.md**: Detailed usage instructions
- **CHANGELOG.md**: This file
- **requirements.txt**: Python dependencies
- **.gitignore**: Proper file exclusions

#### Documentation Features
- **Feature Lists**: Complete capability documentation
- **Usage Examples**: Step-by-step workflows
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Tips for optimal results
- **Keyboard Reference**: Quick shortcut lookup

### 🎯 Enhanced Features Summary

| Category | Before | After |
|----------|--------|-------|
| Annotation Types | 1 (boxes only) | 2 (boxes + polygons) |
| Keyboard Shortcuts | 0 | 13 |
| Navigation | Manual load only | Previous/Next + arrows |
| Zoom Levels | Fixed | 10% - 500% |
| Undo History | None | 50 steps |
| Training Presets | None | 3 presets |
| Progress Visualization | Text only | Bar + console + metrics |
| Dashboard | None | Full statistics tab |
| Import/Export | None | 4 formats each |
| Documentation | None | 4 comprehensive docs |

### 🔮 Future Enhancements

While this version represents a major leap forward, potential future additions could include:
- Multi-language support
- Cloud collaboration features
- Video annotation support
- Auto-annotation AI suggestions
- Model performance metrics dashboard
- Custom export format templates
- Annotation templates/presets
- Team collaboration features
- Advanced data augmentation preview
- Integration with popular ML frameworks

## [1.0.0] - Original Version

### Initial Features
- Basic bounding box annotation
- Simple project management
- Model training with PyTorch
- Screenshot capture
- Image recognition
- OCR sample recognizer

---

**Note**: This changelog reflects the comprehensive enhancement of the application from a basic tool to a professional-grade annotation and training platform suitable for both casual and expert users.
