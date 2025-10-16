# Testing Guide for CUDA Detection Improvements

## Quick Start

### 1. Run the Test Suite

```bash
cd /home/runner/work/Image-rec-labelling-training-ui/Image-rec-labelling-training-ui
python test_features.py
```

Expected output: `32 tests pass` (including 8 new CUDA detection tests)

### 2. Run the CUDA Diagnostics Demo

```bash
python demo_cuda_diagnostics.py
```

This will show:
- Full CUDA diagnostics output
- Device selection testing (auto, force_gpu, force_cpu)
- Summary with troubleshooting steps

### 3. Test the Application UI (requires display)

```bash
python image_recognition.py
```

**What to test:**

1. **Startup Console Output**
   - Look for "Application Startup - CUDA Detection" section
   - Verify CUDA diagnostics are logged
   - Check for troubleshooting steps if CUDA not available

2. **Training Tab**
   - Navigate to "Train" tab
   - Look for device selection section
   - Find "Detected: [CPU/GPU]" label
   - Click "🔍 Check CUDA" button
   - Verify diagnostic dialog opens
   - Click "Copy to Clipboard" button
   - Verify diagnostics copied

3. **Training Process**
   - Start a training session (requires a project with annotations)
   - Check training metrics window
   - Verify "TRAINING STARTED - CUDA Diagnostics" section appears
   - Confirm device selection is logged

4. **Device Selection**
   - Try each device option: Auto, Force GPU, Force CPU
   - Verify warnings appear when appropriate
   - Check status bar shows current device

## Testing Scenarios

### Scenario 1: System with CUDA GPU

**Setup:** System with NVIDIA GPU, CUDA drivers installed, CUDA-enabled PyTorch

**Expected Behavior:**
- CUDA available: True
- Device count > 0
- Device name shows GPU model
- Auto mode uses GPU
- Force GPU mode uses GPU with no warning
- Training logs show GPU device
- Check CUDA button shows "✓ CUDA is working correctly!"

**To Test:**
```bash
# Check GPU is detected
nvidia-smi

# Run diagnostics
python demo_cuda_diagnostics.py

# Expected output includes:
# CUDA available: True
# Device name: [Your GPU Model]
```

### Scenario 2: System with CPU-only PyTorch

**Setup:** System with or without GPU, but PyTorch installed without CUDA

**Expected Behavior:**
- CUDA available: False
- CUDA version: None
- Warning mentions "CPU-only PyTorch installation"
- Auto mode uses CPU
- Force GPU mode uses CPU with warning about reinstalling PyTorch
- Check CUDA button shows link to pytorch.org

**To Test:**
```bash
python demo_cuda_diagnostics.py

# Expected output includes:
# CUDA version: None
# Issue: CPU-only PyTorch installation
# Solution: Reinstall PyTorch with CUDA support
```

### Scenario 3: System with CUDA PyTorch but no GPU/drivers

**Setup:** CUDA-enabled PyTorch installed but no GPU or drivers

**Expected Behavior:**
- CUDA available: False
- CUDA version: [version number, e.g., 12.8]
- Warning mentions driver/GPU issues
- Auto mode uses CPU
- Force GPU mode uses CPU with warning about nvidia-smi
- Check CUDA button suggests checking nvidia-smi and drivers

**To Test:**
```bash
python demo_cuda_diagnostics.py

# Expected output includes:
# CUDA version: 12.8 (or similar)
# CUDA available: False
# Suggests running nvidia-smi
```

### Scenario 4: CUDA_VISIBLE_DEVICES=-1

**Setup:** CUDA system with CUDA_VISIBLE_DEVICES set to hide GPUs

**Expected Behavior:**
- CUDA available: False
- CUDA_VISIBLE_DEVICES: -1
- Warning mentions environment variable
- Check CUDA button shows current variable value
- Suggestions to unset or change variable

**To Test:**
```bash
# Hide GPU
export CUDA_VISIBLE_DEVICES=-1  # Linux/Mac
# or
set CUDA_VISIBLE_DEVICES=-1     # Windows

python demo_cuda_diagnostics.py

# Expected output includes:
# CUDA_VISIBLE_DEVICES: -1
# Troubleshooting mentions environment variable
```

## Unit Test Details

### New Test Classes

**TestCudaDiagnostics** (5 tests)
- `test_get_cuda_diagnostics`: Verifies diagnostic structure
- `test_log_cuda_diagnostics`: Tests logging output
- `test_cuda_not_available_warning`: Checks warning generation
- `test_cpu_only_pytorch_detection`: Mocks CPU-only PyTorch
- `test_cuda_installed_but_unavailable`: Mocks CUDA present but unavailable

**TestEnhancedDeviceSelection** (3 tests)
- `test_force_gpu_with_detailed_warning`: Tests enhanced warnings
- `test_device_always_valid`: Ensures device always returns valid
- `test_warning_messages_user_friendly`: Checks message quality

### Running Specific Tests

```bash
# Run only CUDA diagnostic tests
python -m unittest test_features.TestCudaDiagnostics

# Run only device selection tests
python -m unittest test_features.TestEnhancedDeviceSelection

# Run with verbose output
python test_features.py -v
```

## Manual Verification Checklist

- [ ] Tests pass (32/32)
- [ ] Demo script runs without errors
- [ ] Startup logs show CUDA diagnostics
- [ ] Training logs show CUDA diagnostics
- [ ] Check CUDA button opens dialog
- [ ] Dialog shows correct information for system
- [ ] Copy to clipboard works
- [ ] Device selection changes are saved
- [ ] Warnings appear when Force GPU selected without CUDA
- [ ] Status bar shows current device
- [ ] Training works on CPU when CUDA unavailable
- [ ] Training works on GPU when CUDA available (if applicable)
- [ ] README troubleshooting section is clear
- [ ] No breaking changes to existing functionality

## Debugging Tips

### If Tests Fail

1. **Check PyTorch Installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **Check Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Individual Tests**
   ```bash
   python -m unittest test_features.TestCudaDiagnostics.test_get_cuda_diagnostics -v
   ```

### If CUDA Detection Seems Wrong

1. **Verify with PyTorch Directly**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

2. **Check Environment Variables**
   ```bash
   echo $CUDA_VISIBLE_DEVICES  # Linux/Mac
   echo %CUDA_VISIBLE_DEVICES%  # Windows
   ```

3. **Check GPU with nvidia-smi**
   ```bash
   nvidia-smi
   ```

### If Application Won't Start

- GUI requires tkinter and display
- Use demo script instead: `python demo_cuda_diagnostics.py`
- Check console output for startup errors

## Performance Testing

### Timing Tests

```bash
# Time startup with diagnostics
time python demo_cuda_diagnostics.py

# Should complete in < 1 second
```

### Memory Tests

```bash
# Check memory usage
python -c "
import tracemalloc
tracemalloc.start()

from device_utils import get_cuda_diagnostics, log_cuda_diagnostics
diagnostics = get_cuda_diagnostics()
log_cuda_diagnostics()

current, peak = tracemalloc.get_traced_memory()
print(f'Current memory: {current / 1024 / 1024:.2f} MB')
print(f'Peak memory: {peak / 1024 / 1024:.2f} MB')
tracemalloc.stop()
"

# Should use minimal memory (< 50 MB)
```

## Reporting Issues

When reporting issues with CUDA detection:

1. Run and include output from:
   ```bash
   python demo_cuda_diagnostics.py > cuda_report.txt
   ```

2. Click "Check CUDA" button in app and copy diagnostics

3. Include:
   - Operating system and version
   - GPU model (if applicable)
   - Driver version (from nvidia-smi)
   - PyTorch installation method (pip/conda, CUDA version)
   - Output from demo script

4. Attach to GitHub issue

## Success Criteria

✅ All automated tests pass (32/32)
✅ Demo script runs without errors
✅ CUDA diagnostics appear in startup logs
✅ CUDA diagnostics appear in training logs
✅ Check CUDA button works and shows accurate info
✅ Warnings appear when appropriate
✅ Copy to clipboard works
✅ README is clear and helpful
✅ No breaking changes
✅ Works on systems with and without CUDA
