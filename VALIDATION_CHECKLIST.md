# Compute Device Selection Feature - Validation Checklist

## Pre-Launch Validation

This checklist covers manual validation steps for the compute device selection feature.

### 1. Settings Persistence

- [ ] **Fresh Installation**
  - Delete `config/settings.json` if it exists
  - Launch application
  - Verify `config/settings.json` is created with default values
  - Confirm default device preference is "auto"

- [ ] **Settings Save**
  - Open Training tab
  - Change Compute Device to "Force CPU"
  - Close and reopen application
  - Verify dropdown shows "Force CPU" on restart
  - Verify `config/settings.json` contains `"device_preference": "force_cpu"`

- [ ] **Settings Load**
  - Manually edit `config/settings.json` to set `"device_preference": "force_gpu"`
  - Launch application
  - Verify Training tab dropdown shows "Force GPU"

### 2. UI Components

- [ ] **Compute Device Dropdown**
  - Verify dropdown has three options:
    - "Auto (recommended)"
    - "Force GPU" (only if CUDA available)
    - "Force CPU"
  - Verify default selection is "Auto (recommended)"
  - Verify selections are saved immediately

- [ ] **Detected Device Display**
  - Verify label shows "Detected: [device name]"
  - On CPU system: Should show "Detected: CPU"
  - On GPU system: Should show "Detected: CUDA (GPU Name)"
  - Text color should be gray (#95A5A6) by default

- [ ] **Info Icon**
  - Verify ℹ️ icon appears next to "Compute Device:" label
  - Click on icon
  - Verify notification appears in status bar explaining device options
  - Verify icon color is blue (#3498DB)
  - Verify cursor changes to hand on hover

### 3. Device Selection Logic

- [ ] **Auto Mode**
  - Set dropdown to "Auto (recommended)"
  - Start training
  - On GPU system: Verify training uses GPU
  - On CPU system: Verify training uses CPU
  - Verify no warnings appear

- [ ] **Force CPU Mode**
  - Set dropdown to "Force CPU"
  - Start training
  - Verify training uses CPU (even if GPU available)
  - Verify no warnings appear
  - Verify detected device display still shows actual hardware

- [ ] **Force GPU Mode (GPU Available)**
  - On GPU system only
  - Set dropdown to "Force GPU"
  - Start training
  - Verify training uses GPU
  - Verify no warnings appear

- [ ] **Force GPU Mode (GPU Unavailable)**
  - On CPU-only system or with CUDA disabled
  - Set dropdown to "Force GPU"
  - Start training
  - Verify warning appears: "GPU requested but not available - falling back to CPU"
  - Verify training proceeds on CPU
  - Verify preference remains "Force GPU" for future retry
  - Verify detected device display shows warning in orange color

### 4. Training Logs

- [ ] **Device Logging**
  - Start any training session
  - Check training metrics console
  - Verify logs contain:
    - "Using device: [device name]"
    - "Device preference: [auto/force_gpu/force_cpu]"
  - Verify device name matches actual device used

### 5. Status Bar

- [ ] **Device Indicator**
  - Check bottom-left of main window
  - Verify "🖥️ Device: [name]" appears
  - Verify it reflects actual hardware detection
  - Verify it updates if system state changes

### 6. Error Handling

- [ ] **Corrupted Settings File**
  - Edit `config/settings.json` to contain invalid JSON
  - Launch application
  - Verify application loads with default settings
  - Verify no crash occurs

- [ ] **Missing Settings File**
  - Delete `config/settings.json`
  - Launch application
  - Verify application creates new settings file
  - Verify default values are used

- [ ] **Invalid Preference Value**
  - Edit `config/settings.json`: set `"device_preference": "invalid_value"`
  - Launch application
  - Verify application handles gracefully (treats as "auto")

### 7. Cross-Platform Compatibility

- [ ] **Windows**
  - Test all device options
  - Verify GPU detection works with NVIDIA GPUs
  - Verify CPU fallback works

- [ ] **Linux**
  - Test all device options
  - Verify CUDA paths are handled correctly
  - Verify CPU fallback works

- [ ] **macOS**
  - Verify CPU mode works (no CUDA on macOS)
  - Verify "Force GPU" shows appropriate warning

### 8. Integration Tests

- [ ] **End-to-End Training**
  - Create a test project with 10+ annotated images
  - Test training with each device option:
    - Auto (recommended)
    - Force GPU (if available)
    - Force CPU
  - Verify all complete successfully
  - Verify model files are created
  - Verify trained model can be used for recognition

- [ ] **Settings Across Sessions**
  - Set preference to "Force CPU"
  - Train a model
  - Close application
  - Reopen application
  - Verify preference is still "Force CPU"
  - Train another model
  - Verify it uses CPU

### 9. Documentation Verification

- [ ] **USER_GUIDE.md**
  - Verify compute device section exists
  - Verify all options are documented
  - Verify troubleshooting section covers GPU issues

- [ ] **SETTINGS.md**
  - Verify complete settings documentation exists
  - Verify examples are correct
  - Verify API reference is accurate

- [ ] **README.md**
  - Verify feature is listed in version 3.0 features
  - Verify description is accurate

### 10. Automated Tests

- [ ] **Run Test Suite**
  ```bash
  python test_features.py
  ```
  - Verify all 24 tests pass
  - Verify TestDevicePreferences tests pass
  - No errors or warnings

### 11. Performance

- [ ] **GPU Performance**
  - On GPU system, train with Force GPU
  - Note training time
  - Train same project with Force CPU
  - Verify GPU is 2-10x faster

- [ ] **No Performance Regression**
  - Train a model with new code
  - Compare time to previous version
  - Verify no significant slowdown in any mode

## Bug Fixes & Edge Cases

### Known Limitations
- GPU detection only works with CUDA-compatible NVIDIA GPUs
- AMD GPUs are not currently supported (treated as CPU)
- Apple Silicon (M1/M2) GPU acceleration not yet supported

### Future Enhancements
- Add AMD ROCm support
- Add Apple Metal (MPS) support
- Add device memory usage display
- Add automatic device recommendation based on model size

## Sign-Off

- [ ] All validation items completed
- [ ] All automated tests passing
- [ ] Documentation complete and accurate
- [ ] No critical bugs found
- [ ] Feature ready for production

**Validated By:** _________________  
**Date:** _________________  
**Environment:** _________________  
**GPU Info:** _________________
