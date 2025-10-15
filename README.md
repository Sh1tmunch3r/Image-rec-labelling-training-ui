# Image Labeling Studio Pro 🎨

A professional, full-featured image annotation and model training application with modern UI/UX designed for both casual and expert users.

## ✨ Features

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
2. Choose a preset or customize hyperparameters
3. Click "🚀 Start Training"
4. Monitor real-time progress and metrics

### 5. Use Your Model
1. Go to the "Recognize" tab
2. Select your trained model
3. Capture or load an image
4. View recognition results

### 6. Monitor Progress
- Check the "Dashboard" tab for project statistics
- View class distribution
- Export reports
- Validate annotations

## 🎨 UI/UX Improvements

### Professional Design
- Modern color scheme with dark theme
- Intuitive icon usage
- Clear visual hierarchy
- Responsive panel layouts
- Smooth interactions

### User Experience
- Minimal clicks to common actions
- Keyboard-first workflow support
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
