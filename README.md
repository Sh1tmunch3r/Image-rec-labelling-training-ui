# Image Labeling Studio Pro 🎨

## Version 2.0 - Now with Intelligent Auto-Training & Smart Detection! 🚀

A professional, full-featured image annotation and model training application with modern UI/UX, intelligent parameter selection, and advanced detection capabilities designed for both casual and expert users.

## ✨ NEW in Version 3.0

### 🎥 Live Recognition Mode
- **Real-time screen capture** - Continuously capture and recognize at configurable FPS (1-10)
- **Live preview** - See detections update in real-time as screen content changes
- **Adjustable frame rate** - Balance between responsiveness and system load
- **Batch frame capture** - Save multiple frames with annotations at once
- **Seamless toggle** - Switch between live mode and single-shot capture

### 💾 Advanced Export with Annotations
- **COCO JSON format** - Industry-standard format ready for training pipelines
- **Per-image JSON** - Simple format for individual file processing
- **Complete metadata** - Saves bounding boxes, labels, confidence scores, timestamps
- **Organized structure** - Automatic folder organization (images/ and annotations/)
- **Training-ready** - Exports work directly with common training frameworks

### 🖥️ GPU Auto-Detection & Training
- **Automatic CUDA detection** - Detects and uses GPU automatically at startup
- **Smart fallback** - Gracefully falls back to CPU if GPU initialization fails
- **Device override** - Force CPU or GPU usage via Settings
- **Real-time indicator** - Status bar shows current device (GPU/CPU)
- **5-10x faster training** - Leverage GPU acceleration when available

### 🧠 Intelligent Auto-Training
- **One-click optimization** - Automatically configures training parameters based on your dataset
- **Smart analysis** - Considers dataset size, annotations per image, and number of classes
- **Detailed explanations** - Shows reasoning behind each setting choice
- **Adaptive learning** - Adjusts epochs, learning rate, and batch size intelligently

### 🎯 Smart Detection Control
- **Adjustable confidence threshold** - Interactive slider from 0.1 to 0.95
- **Non-Maximum Suppression (NMS)** - Removes duplicate detections automatically
- **One box per object** - Clean, professional results
- **Real-time filtering** - Instant updates when threshold changes

### 📊 Enhanced Status Bar
- **Device information** - Shows active training device (GPU/CPU)
- **Live notifications** - Real-time feedback for all operations
- **Status messages** - Clear indicators for saves, errors, and warnings
- **Color-coded alerts** - Visual distinction between info, success, warning, and error

### 📚 Comprehensive Help System
- **F1 Quick Help** - Press F1 anywhere for instant guidance
- **Training Guide** - Complete explanation of all parameters
- **Keyboard Shortcuts** - Easy reference for all commands
- **Professional documentation** - Clear, organized, accessible

### 🎯 Enhanced Labeling
- **Auto-save functionality** - Saves annotations automatically when navigating
- **Real-time size preview** - Shows dimensions during box drawing
- **Enhanced visuals** - Bright colors and better contrast for drawing
- **Professional polish** - Modern UI with emoji icons and rounded corners

## ✨ Core Features

### 🎯 Advanced Annotation Tools
- **Multiple Annotation Types**
  - Bounding boxes with drag-and-drop
  - Polygon annotations for precise shapes
  - Multi-class label support
  - Real-time visual feedback

### ⌨️ Keyboard Shortcuts
- `Ctrl+S`: Save annotations
- `Ctrl+Z/Y`: Undo/Redo
- `B`: Switch to box annotation mode
- `P`: Switch to polygon annotation mode
- `Delete`: Remove selected annotation
- `←/→`: Navigate between images
- `Ctrl +/-`: Zoom in/out
- `Ctrl+0`: Reset zoom
- `Ctrl+C/V`: Copy/Paste annotations

### 🔍 Image Navigation & Viewing
- **Zoom & Pan Controls**
  - Mouse wheel zoom
  - Middle-click pan
  - Keyboard zoom shortcuts
  - Reset to 100% view
- **Image Navigation**
  - Next/Previous image buttons
  - Arrow key navigation
  - Image counter display
  - Batch image loading

### 📊 Project Management Dashboard
- **Real-time Statistics**
  - Total images count
  - Annotated images progress
  - Total annotations
  - Class distribution visualization
- **Quick Actions**
  - Refresh statistics
  - Validate annotations
  - Export reports
  - Backup projects

### 💾 Import/Export
- **Multiple Format Support**
  - COCO JSON
  - YOLO TXT
  - Pascal VOC XML
  - CSV
- **Batch Operations**
  - Copy/paste annotations between images
  - Bulk export
  - Project backup

### 🚀 Advanced Model Training
- **Intelligent Auto-Configuration** ⭐ NEW
  - One-click optimization based on dataset size
  - Automatic parameter selection
  - Detailed reasoning and explanations
  - Adapts to dataset complexity
- **Hyperparameter Tuning**
  - Configurable epochs
  - Learning rate control
  - Batch size optimization
  - Momentum and weight decay
  - Data augmentation options
- **Training Presets**
  - Fast (5 epochs)
  - Balanced (10 epochs, default)
  - Accurate (20 epochs)
- **Real-time Progress**
  - Visual progress bar
  - Loss metrics display
  - Epoch-by-epoch logging
  - Training console output

### ♿ Accessibility & UX
- **Onboarding Experience**
  - Welcome dialog with tutorial
  - Keyboard shortcuts guide
  - Quick start instructions
- **Professional UI**
  - Modern dark theme
  - Responsive layout
  - Resizable panels
  - Color-coded annotations
  - Status indicators
  - Emoji-enhanced buttons

### 🔄 Undo/Redo System
- Full annotation history
- 50-step undo buffer
- Redo support
- State preservation

### ✅ Validation & Quality Control
- Annotation validation
- JSON integrity checks
- Error reporting
- Statistics tracking

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Sh1tmunch3r/Image-rec-labelling-training-ui.git
cd Image-rec-labelling-training-ui

# Install dependencies
pip install -r requirements.txt
```

### Optional: Tesseract OCR
For text recognition features, install Tesseract OCR:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

### Running the Application
```bash
python image_recognition.py
```

## 📖 Usage Guide

### 1. Create a New Project
1. Click "New Project" in the top bar
2. Enter a project name
3. Add your classes using "Add Class" button

### 2. Load Images
- **Capture**: Take a screenshot of your screen
- **Load**: Import images from files
- Navigate with arrow keys or buttons

### 3. Annotate Images
- **Box Mode (B)**: Click and drag to create bounding boxes
- **Polygon Mode (P)**: Click to add points, right-click to finish
- Select or create labels for each annotation
- Use Ctrl+S to save

### 4. Train Your Model
1. Go to the "Train" tab
2. **Click "🧠 Auto-Configure Settings"** for intelligent optimization ⭐ NEW
   - Or choose a preset or customize hyperparameters manually
3. Click "🚀 Start Training"
4. Monitor real-time progress and metrics

### 5. Use Your Model
1. Go to the "Recognize" tab
2. Select your trained model
3. **Adjust confidence threshold** for desired detection sensitivity ⭐ NEW
4. **Enable NMS** to remove duplicate detections ⭐ NEW
5. Choose between:
   - Single capture: Click "Capture & Recognize" for one-time detection
   - **Live mode**: Enable "Live Recognition" for continuous monitoring ⭐ NEW v3.0
6. **Save images with annotations** in COCO JSON or per-image format ⭐ NEW v3.0
7. View clean, filtered recognition results

### 6. Monitor Progress
- Check the "Dashboard" tab for project statistics
- View class distribution
- Export reports
- Validate annotations

## 🎨 UI/UX Improvements

### Professional Design (Enhanced in v2.0)
- Modern dark theme with optimized contrast
- **Emoji icons** throughout for visual guidance ⭐ NEW
- **Rounded corners** on all buttons (8-10px) ⭐ NEW
- **Consistent sizing** - all buttons 32-40px height ⭐ NEW
- Clear visual hierarchy
- Responsive panel layouts
- Smooth interactions
- **Enhanced canvas** with professional background ⭐ NEW

### User Experience
- Minimal clicks to common actions
- Keyboard-first workflow support
- **Auto-save on navigation** ⭐ NEW
- **Real-time size preview** while drawing ⭐ NEW
- **F1 quick help** from anywhere ⭐ NEW
- Visual feedback for all actions
- Status indicators everywhere
- Error prevention and recovery

### Performance
- Lazy image loading
- Efficient canvas rendering
- Optimized for large datasets
- Background training thread

## 🔧 Configuration

### Training Presets
- **Fast**: Quick iterations, lower accuracy
- **Balanced**: Good trade-off (default)
- **Accurate**: Best quality, longer training

### Customization
All hyperparameters can be manually adjusted:
- Epochs (1-100+)
- Learning Rate (0.0001-0.1)
- Batch Size (1-32)
- Momentum (0-1)
- Weight Decay (0-0.01)

## 📊 Project Structure
```
Image-rec-labelling-training-ui/
├── image_recognition.py      # Main application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── recognizers/               # Trained model recognizers
│   └── sample_text.py        # Sample OCR recognizer
└── projects/                  # User projects
    └── [project_name]/
        ├── images/           # Source images
        ├── annotations/      # JSON annotations
        ├── classes.txt       # Class definitions
        └── model.pth        # Trained model
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📝 License
This project is open source and available for use.

## 🙏 Acknowledgments
- Built with CustomTkinter for modern UI
- PyTorch for deep learning
- Pillow for image processing
- MSS for screen capture

## 📧 Support
For issues, questions, or feature requests, please open an issue on GitHub.

---

**Image Labeling Studio Pro** - Transform your image labeling workflow! 🚀
