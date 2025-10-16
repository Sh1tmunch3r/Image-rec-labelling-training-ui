# Application Settings Documentation

## Overview
Image Labeling Studio Pro stores application settings in a JSON configuration file located at `config/settings.json`. Settings are automatically loaded at startup and saved when changed through the UI.

## Settings File Location
```
Image-rec-labelling-training-ui/
├── config/
│   └── settings.json
```

## Settings File Format

### Structure
```json
{
  "device_preference": "auto",
  "version": "1.0"
}
```

### Available Settings

#### `device_preference`
Controls which compute device is used for model training.

**Type:** String

**Valid Values:**
- `"auto"` (default) - Automatically uses GPU if available, otherwise CPU
- `"force_gpu"` - Always attempts to use GPU, falls back to CPU if unavailable with a warning
- `"force_cpu"` - Always uses CPU for training

**Example:**
```json
{
  "device_preference": "force_cpu"
}
```

**UI Control:** Training tab → Compute Device dropdown

**Behavior:**
- When set to `"auto"`:
  - Application detects CUDA availability
  - Uses GPU if `torch.cuda.is_available()` returns true
  - Falls back to CPU silently if no GPU detected
  
- When set to `"force_gpu"`:
  - Attempts to initialize CUDA device
  - Shows warning notification if GPU unavailable
  - Falls back to CPU but keeps user preference saved
  - Useful for troubleshooting GPU issues
  
- When set to `"force_cpu"`:
  - Always uses CPU regardless of GPU availability
  - Useful for testing, debugging, or when GPU is unstable
  - No performance warnings shown

#### `version`
Settings file format version for future compatibility.

**Type:** String

**Current Value:** `"1.0"`

**Note:** This field is reserved for future use to handle settings migration if the format changes in future versions.

## Modifying Settings

### Through the UI (Recommended)
1. Open the application
2. Navigate to the **Train** tab
3. Find the **Compute Device** dropdown
4. Select your preferred option:
   - Auto (recommended)
   - Force GPU
   - Force CPU
5. Changes are saved immediately

### Manual Editing
You can also edit the `config/settings.json` file directly:

1. Close the application
2. Open `config/settings.json` in a text editor
3. Modify the values
4. Save the file
5. Restart the application

**Warning:** Ensure the JSON format is valid. Invalid JSON will be ignored and default settings will be used.

## Default Settings
If the settings file doesn't exist or is corrupted, the application uses these defaults:
```json
{
  "device_preference": "auto",
  "version": "1.0"
}
```

## Settings Persistence
- Settings are saved immediately when changed through the UI
- Settings persist across application restarts
- Settings are stored per installation (not per project)
- If settings file is deleted, defaults are restored automatically

## Troubleshooting

### Settings not saving
1. Check file permissions on the `config` folder
2. Ensure the application has write access to the directory
3. Look for error messages in the console/terminal

### Settings reset to defaults
1. Verify `config/settings.json` exists
2. Check if the file contains valid JSON
3. Review file permissions (should be readable/writable)

### GPU selection not working
1. Verify `device_preference` is set correctly in settings.json
2. Check CUDA availability by running:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. Ensure PyTorch with CUDA support is installed
4. Restart application after changing settings

## Advanced Configuration

### Future Settings
The settings system is designed to be extensible. Future versions may add:
- UI theme preferences
- Default project paths
- Auto-save intervals
- Keyboard shortcut customization
- Model training defaults

### Settings Schema Validation
The application validates settings on load:
- Invalid keys are ignored
- Invalid values fallback to defaults
- Missing required fields use defaults
- Unknown fields are preserved for forward compatibility

## API Reference

### Functions in `device_utils.py`

#### `load_settings()`
Loads settings from the config file.

**Returns:** Dictionary with settings

**Example:**
```python
from device_utils import load_settings
settings = load_settings()
print(settings['device_preference'])
```

#### `save_settings(settings)`
Saves settings to the config file.

**Parameters:**
- `settings` (dict): Dictionary containing settings to save

**Returns:** Boolean indicating success

**Example:**
```python
from device_utils import save_settings
settings = {
    "device_preference": "force_cpu",
    "version": "1.0"
}
success = save_settings(settings)
```

#### `get_device(preference='auto')`
Gets PyTorch device based on preference.

**Parameters:**
- `preference` (str): Device preference ('auto', 'force_gpu', or 'force_cpu')

**Returns:** Tuple of (torch.device, device_name, warning_message)

**Example:**
```python
from device_utils import get_device
device, name, warning = get_device('auto')
print(f"Using: {name}")
if warning:
    print(f"Warning: {warning}")
```

## Version History

### Version 1.0 (Current)
- Initial settings system
- Device preference support
- Auto-detection and fallback logic
