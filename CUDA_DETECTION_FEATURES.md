# CUDA Detection and Device Selection Features

## Overview

This document describes the comprehensive CUDA detection and device selection improvements implemented in the Image Labeling Studio Pro application.

## Features Implemented

### 1. Startup CUDA Diagnostics

When the application starts, it automatically logs comprehensive CUDA diagnostics to the console:

```
============================================================
Application Startup - CUDA Detection
============================================================
============================================================
CUDA Diagnostics
============================================================
PyTorch version: 2.9.0+cu128
CUDA version: 12.8
CUDA available: False
CUDA device count: 0
CUDA_VISIBLE_DEVICES: Not set
Python executable: /usr/bin/python
============================================================

⚠️ CUDA NOT DETECTED - Troubleshooting:

  • PyTorch has CUDA support but cannot detect GPU
  • Possible causes:
    - No NVIDIA GPU in system
    - NVIDIA drivers not installed or outdated
    - CUDA_VISIBLE_DEVICES is hiding GPUs
    - Incompatible CUDA/driver version
  • Verification steps:
    1. Run 'nvidia-smi' to check GPU and driver
    2. Check NVIDIA driver version matches CUDA requirements
    3. Ensure CUDA_VISIBLE_DEVICES is not set to -1 or empty

  Training will use CPU (slower but functional)
============================================================
```

### 2. Training Time CUDA Logging

When training starts, comprehensive CUDA diagnostics are logged to the training metrics display:

```
============================================================
TRAINING STARTED - CUDA Diagnostics
============================================================
PyTorch version: 2.9.0+cu128
CUDA version: 12.8
CUDA available: False
CUDA device count: 0
CUDA_VISIBLE_DEVICES: Not set
============================================================
Starting training with 10 epochs
Learning rate: 0.005, Batch size: 2
Using device: CPU (preference: auto)
--------------------------------------------------
```

### 3. "Check CUDA" Button in Training Tab

**Location:** Training tab → Device selection section → "🔍 Check CUDA" button

**Features:**
- Displays comprehensive CUDA system diagnostics in a popup dialog
- Shows PyTorch version, CUDA version, device information
- Provides specific troubleshooting steps based on detected issues
- Includes "Copy to Clipboard" button for easy bug reporting
- User-friendly explanations with actionable solutions

**Dialog Content:**
```
============================================================
SYSTEM INFORMATION
============================================================
PyTorch version: 2.9.0+cu128
CUDA version: 12.8
Python executable: /usr/bin/python

============================================================
CUDA STATUS
============================================================
CUDA available: False
CUDA device count: 0
CUDA_VISIBLE_DEVICES: Not set

⚠️ CUDA NOT DETECTED

============================================================
TROUBLESHOOTING STEPS
============================================================

Issue: CUDA not detected despite PyTorch CUDA support

Possible causes:

1. No NVIDIA GPU in system
   - Check if you have an NVIDIA GPU
   - Run 'nvidia-smi' in terminal to verify

2. NVIDIA drivers not installed or outdated
   - Download latest drivers from nvidia.com
   - Your PyTorch expects CUDA 12.8
   - Driver version must support this CUDA version

3. CUDA_VISIBLE_DEVICES environment variable issue
   - Current value: Not set
   - If set to -1 or empty, GPUs are hidden
   - Unset it or set to valid GPU IDs (0,1,2...)

4. Incompatible CUDA/driver version
   - Run 'nvidia-smi' to check driver version
   - Ensure driver supports CUDA version in PyTorch

Verification commands:
  nvidia-smi          # Check GPU and driver
  nvcc --version      # Check CUDA toolkit

Note: Training will use CPU (slower but functional)

============================================================
```

### 4. Enhanced Device Selection Warnings

When user selects "Force GPU" but CUDA is not available:

**Status Bar Notification:**
```
⚠️ GPU requested but CUDA not available. Check drivers with nvidia-smi
```

**For CPU-only PyTorch:**
```
⚠️ GPU requested but your PyTorch is CPU-only. Reinstall PyTorch with CUDA: https://pytorch.org
```

### 5. Device Selection UI Elements

**Training Tab - Device Selection:**
- Radio buttons: "Auto (recommended)", "Force GPU" (if CUDA available), "Force CPU"
- Detected device display: Shows current device (e.g., "Detected: CPU")
- Info tooltip: Explains each device selection mode
- Check CUDA button: Opens full diagnostics dialog

**Status Bar:**
- Shows active device: "🖥️ Device: CPU" or "🖥️ Device: CUDA (GeForce GTX 1650)"

## Device Selection Logic

### Auto Mode (Default)
- Automatically uses CUDA if available
- Falls back to CPU if CUDA not available
- No warnings shown
- Recommended for most users

### Force GPU Mode
- Always tries to use CUDA
- Falls back to CPU if CUDA not available
- Shows detailed warning with troubleshooting steps
- Useful for users who expect GPU but want to be notified if it's not working

### Force CPU Mode
- Always uses CPU even if CUDA available
- No warnings
- Useful for testing or when GPU is needed for other tasks

## Code Architecture

### device_utils.py

**New Functions:**
- `get_cuda_diagnostics()`: Collects comprehensive CUDA information
- `log_cuda_diagnostics(logger_func)`: Logs diagnostics with troubleshooting
- `get_device(preference)`: Enhanced with detailed warnings

### image_recognition.py

**Enhanced Methods:**
- `detect_device()`: Logs diagnostics at startup
- `show_cuda_diagnostics()`: UI dialog for CUDA diagnostics
- `train_model()`: Logs diagnostics at training start
- `update_detected_device_display()`: Shows warnings in UI

**New UI Elements:**
- Check CUDA button in Training tab
- Copy to clipboard functionality in diagnostics dialog

## Testing

### Test Coverage

**TestCudaDiagnostics:**
- `test_get_cuda_diagnostics`: Verifies diagnostic data structure
- `test_log_cuda_diagnostics`: Verifies logging output
- `test_cuda_not_available_warning`: Tests warning messages
- `test_cpu_only_pytorch_detection`: Tests CPU-only PyTorch detection
- `test_cuda_installed_but_unavailable`: Tests CUDA present but not available

**TestEnhancedDeviceSelection:**
- `test_force_gpu_with_detailed_warning`: Tests enhanced warnings
- `test_device_always_valid`: Ensures device always valid
- `test_warning_messages_user_friendly`: Verifies user-friendly messages

### Running Tests

```bash
python test_features.py
```

All 32 tests pass, including 8 new tests for CUDA detection.

## User Documentation

Comprehensive troubleshooting documentation added to README.md:
- Common CUDA issues and solutions
- Verification commands
- Step-by-step fixes for various scenarios
- Performance notes

## Compatibility

- No breaking changes
- Works with both CUDA and non-CUDA systems
- Graceful fallback to CPU when CUDA unavailable
- Settings preserved across sessions

## Performance Impact

- Minimal overhead at startup (< 100ms)
- No impact on training performance
- Diagnostics button triggers on-demand only

## Benefits

1. **User-Friendly**: Clear explanations for non-technical users
2. **Actionable**: Specific steps to resolve issues
3. **Comprehensive**: Covers all common CUDA problems
4. **Debuggable**: Easy to share diagnostics for support
5. **Robust**: Handles all edge cases gracefully
6. **Educational**: Helps users understand CUDA/GPU setup

## Example Use Cases

### Use Case 1: User has GPU but CPU-only PyTorch
**Symptom:** Application uses CPU despite having GPU
**Detection:** CUDA version shows as None
**Solution:** Check CUDA button provides PyTorch reinstall instructions with link

### Use Case 2: User has CUDA PyTorch but outdated drivers
**Symptom:** CUDA available = False but version present
**Detection:** CUDA version shows (e.g., 12.8) but not available
**Solution:** Check CUDA button suggests checking nvidia-smi and updating drivers

### Use Case 3: User has everything but CUDA_VISIBLE_DEVICES=-1
**Symptom:** GPU hidden from PyTorch
**Detection:** Environment variable shown in diagnostics
**Solution:** Check CUDA button shows variable value and how to fix it

## Demo Script

A standalone demo script is provided:

```bash
python demo_cuda_diagnostics.py
```

This demonstrates all CUDA detection features without requiring the GUI.
