# Changelog - Version 3.0

## Version 3.0 - Advanced Recognition & GPU Acceleration (2025-10-16)

### 🎥 Live Recognition Mode
**Real-time screen monitoring with continuous detection**

- **Live Recognition Toggle**: Enable continuous screen capture and detection at configurable frame rates
- **Adjustable FPS**: Set capture rate from 1-10 FPS to balance performance and responsiveness
- **Live Preview**: See detections update in real-time as screen content changes
- **Batch Frame Capture**: Save multiple frames with their annotations in one operation
- **Configurable Duration**: Set number of frames to capture when saving in live mode
- **Seamless Mode Switching**: Toggle between live mode and single-shot capture without restart

#### Usage
1. Go to Recognize tab
2. Check "Enable Live Recognition" to start continuous monitoring
3. Adjust FPS slider (default: 3 FPS)
4. Click "Save Images + Annotations" to capture multiple frames
5. Uncheck to return to single-shot mode

---

### 💾 Save Annotations with Images
**Export detected images with complete annotation metadata**

- **COCO JSON Format**: Industry-standard format compatible with major training frameworks
  - Complete dataset structure with images, annotations, and categories
  - Proper bbox format [x, y, width, height]
  - Category mappings and IDs
  - Ready for direct use in training pipelines
  
- **Per-Image JSON Format**: Simple, human-readable format for individual processing
  - One JSON file per image
  - Complete detection metadata per file
  - Easy to parse and process

- **Organized Export Structure**:
  ```
  exports/
  └── recognition_YYYY-MM-DD_HH-MM-SS/
      ├── images/
      │   ├── detection_001.png
      │   ├── detection_002.png
      │   └── ...
      └── annotations/
          ├── instances.json (COCO format)
          or
          ├── detection_001.json (per-image format)
          ├── detection_002.json
          └── ...
  ```

- **Complete Metadata Saved**:
  - Bounding box coordinates
  - Class labels
  - Confidence scores
  - Timestamps
  - Source image information
  - Image dimensions

- **Edge Case Handling**:
  - Graceful handling of zero detections
  - Duplicate filename prevention
  - Permission error handling
  - User feedback on success/failure

#### Usage
1. Run detection (single or live mode)
2. Select format: "COCO JSON" or "Per-image JSON"
3. Click "💾 Save Images + Annotations"
4. Images and annotations saved to `exports/recognition_[timestamp]/`

---

### 🖥️ GPU Auto-Detection & Acceleration
**Automatic CUDA detection with intelligent fallback**

- **Automatic Device Detection**: Detects CUDA/GPU availability on startup
- **Smart Fallback**: Gracefully falls back to CPU if GPU initialization fails
- **Device Override Options**:
  - Auto: Automatically use best available device (default)
  - Force CPU: Use CPU even if GPU available
  - Force GPU: Use GPU even if auto-detection suggests CPU

- **Real-Time Device Indicator**: Status bar shows current device
  - Shows "GPU (NVIDIA GeForce XXX)" when using CUDA
  - Shows "CPU" when using CPU
  - Shows "CPU (fallback)" if GPU failed to initialize

- **Robust Error Handling**:
  - Catches CUDA initialization errors
  - Provides clear error messages with debugging steps
  - Continues operation on CPU without crashing

- **Training Performance**:
  - 5-10x faster training on GPU
  - Automatic tensor placement on correct device
  - Optimized memory usage

#### Technical Details
- Uses `torch.cuda.is_available()` for detection
- Tests actual tensor creation before committing to device
- Logs device information to training console
- All model operations automatically use selected device

#### Usage
1. App automatically detects GPU on startup
2. Check status bar for current device
3. Go to Train tab → Device dropdown to override
4. Training automatically uses selected/detected device

---

### 📊 Enhanced Status Bar
**Real-time feedback and system information**

- **Device Information Display**: Shows active GPU/CPU device
- **Live Status Messages**: Real-time updates for all operations
- **Color-Coded Notifications**:
  - 🔵 Blue: Information
  - 🟢 Green: Success
  - 🟠 Orange: Warning
  - 🔴 Red: Error

- **Operation Feedback**:
  - "Live recognition running — 3 FPS"
  - "Saved 12 images and annotations to ./exports/..."
  - "Training started on GPU (NVIDIA GeForce RTX 3090)"
  - "GPU initialization failed, using CPU"

---

### 🎨 UI/UX Improvements

- **Recognize Tab Redesign**:
  - Reorganized controls with clear sections
  - Live mode controls grouped together
  - Export options prominent and accessible
  - Format selector easily visible

- **Improved Button Styling**:
  - Enhanced "Save Images + Annotations" button (green, bold)
  - Clear visual hierarchy
  - Consistent sizing and spacing
  - Hover states for all interactive elements

- **Better Visual Feedback**:
  - Live FPS indicator updates in real-time
  - Frame count input for batch capture
  - Format selector for export options
  - Status messages for all operations

---

### 🧪 Comprehensive Testing

Added `test_features.py` with 18 unit tests covering:

- **Device Detection Tests** (4 tests):
  - CUDA availability detection
  - CPU device creation
  - CUDA device creation (when available)
  - Device fallback on errors

- **Annotation Format Tests** (4 tests):
  - COCO JSON structure validation
  - Per-image JSON structure validation
  - Empty detection handling
  - COCO bbox format conversion

- **Live Recognition Tests** (3 tests):
  - FPS range validation
  - Frame capture interval calculation
  - Frames to save parameter validation

- **Image Saving Tests** (3 tests):
  - Image save and load
  - Duplicate filename handling
  - Export directory structure creation

- **NMS Filtering Tests** (2 tests):
  - IoU calculation
  - Confidence threshold filtering

- **Backwards Compatibility Tests** (2 tests):
  - Default settings validation
  - Existing annotation format support

Run tests with: `python3 test_features.py`

---

### 📝 Documentation Updates

- **Updated README.md**:
  - Version 3.0 feature highlights
  - Live recognition usage guide
  - Export format documentation
  - GPU/CPU device selection guide

- **Updated Help Dialog (F1)**:
  - Live recognition mode explanation
  - Save annotations format details
  - Device selection guide
  - FPS and performance tips

- **This CHANGELOG**:
  - Complete feature documentation
  - Usage examples
  - Technical implementation details
  - Migration notes

---

### 🔄 Backwards Compatibility

**All existing features remain fully functional:**

- Existing projects load without modification
- Old annotation format still supported
- Default settings preserve v2.0 behavior
- New features are opt-in
- No breaking changes to API or file formats

**Configuration Defaults:**
- Live mode: OFF (use single capture)
- Device: Auto (detect GPU, fallback to CPU)
- Export format: COCO JSON
- FPS: 3 (when live mode enabled)
- Frames to save: 10 (when live mode enabled)

---

### 🐛 Bug Fixes

- Fixed potential race condition in live recognition thread
- Improved error handling for file save operations
- Better handling of empty detection lists
- More robust device detection and fallback

---

### ⚡ Performance Improvements

- GPU training 5-10x faster when available
- Optimized live capture loop for minimal latency
- Efficient COCO JSON generation for large datasets
- Reduced memory usage in batch frame capture

---

### 🔧 Technical Changes

**New Dependencies:**
- No additional dependencies required
- Uses existing torch, torchvision, PIL, numpy, mss

**New Files:**
- `test_features.py`: Comprehensive test suite
- `CHANGELOG_v3.md`: This changelog

**Modified Files:**
- `image_recognition.py`: Core application with all new features
- `README.md`: Updated documentation
- `.gitignore`: Added exports/ directory

**New Directories:**
- `exports/`: Auto-created for recognition exports (gitignored)

---

### 📋 Migration Guide

**From v2.0 to v3.0:**

1. **No action required for basic usage** - all v2.0 features work unchanged

2. **To use GPU acceleration:**
   - Ensure CUDA is installed (if you have NVIDIA GPU)
   - App will auto-detect and use GPU automatically
   - Check status bar to confirm device

3. **To use live recognition:**
   - Go to Recognize tab
   - Check "Enable Live Recognition"
   - Adjust FPS as desired
   - Click "Save Images + Annotations" to capture frames

4. **To export with annotations:**
   - Run recognition (single or live)
   - Select format (COCO JSON recommended for training)
   - Click "Save Images + Annotations"
   - Find exports in `exports/recognition_[timestamp]/`

5. **For training pipelines:**
   - Use COCO JSON format for compatibility
   - Structure is: `exports/[run]/images/` and `exports/[run]/annotations/instances.json`
   - Directly compatible with Detectron2, MMDetection, etc.

---

### 🙏 Acknowledgments

Built on top of Image Labeling Studio Pro v2.0 with:
- PyTorch for deep learning and CUDA support
- MSS for efficient screen capture
- CustomTkinter for modern UI
- Pillow for image processing

---

### 📞 Support

For issues, questions, or feature requests:
- GitHub Issues: [https://github.com/Sh1tmunch3r/Image-rec-labelling-training-ui/issues](https://github.com/Sh1tmunch3r/Image-rec-labelling-training-ui/issues)
- Press F1 in-app for help
- Check README.md for usage guides

---

**Image Labeling Studio Pro v3.0** - Professional annotation, live recognition, and GPU-accelerated training! 🚀
