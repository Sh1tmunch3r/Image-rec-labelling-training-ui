"""
Device selection utilities for training
Separated from main app to allow testing without GUI dependencies
"""
import torch
import json
import os
import sys

CONFIG_FOLDER = "config"
SETTINGS_FILE = os.path.join(CONFIG_FOLDER, "settings.json")


def load_settings():
    """Load application settings from config file"""
    if not os.path.exists(CONFIG_FOLDER):
        os.makedirs(CONFIG_FOLDER)
    
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    # Return default settings
    return {
        "device_preference": "auto",
        "version": "1.0"
    }


def save_settings(settings):
    """Save application settings to config file"""
    if not os.path.exists(CONFIG_FOLDER):
        os.makedirs(CONFIG_FOLDER)
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def get_cuda_diagnostics():
    """
    Get comprehensive CUDA diagnostic information.
    
    Returns:
        dict: Dictionary with CUDA diagnostic information
    """
    diagnostics = {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        'python_executable': sys.executable,
    }
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            diagnostics['device_name'] = torch.cuda.get_device_name(0)
        except Exception as e:
            diagnostics['device_name'] = f"Error getting device name: {e}"
    
    return diagnostics


def log_cuda_diagnostics(logger_func=print):
    """
    Log CUDA diagnostics information.
    
    Args:
        logger_func: Function to use for logging (default: print)
    """
    diagnostics = get_cuda_diagnostics()
    
    logger_func("=" * 60)
    logger_func("CUDA Diagnostics")
    logger_func("=" * 60)
    logger_func(f"PyTorch version: {diagnostics['torch_version']}")
    logger_func(f"CUDA version: {diagnostics['cuda_version']}")
    logger_func(f"CUDA available: {diagnostics['cuda_available']}")
    logger_func(f"CUDA device count: {diagnostics['device_count']}")
    
    if diagnostics['device_name']:
        logger_func(f"Device name: {diagnostics['device_name']}")
    
    logger_func(f"CUDA_VISIBLE_DEVICES: {diagnostics['cuda_visible_devices']}")
    logger_func(f"Python executable: {diagnostics['python_executable']}")
    logger_func("=" * 60)
    
    # Provide troubleshooting if CUDA not available
    if not diagnostics['cuda_available']:
        logger_func("")
        logger_func("⚠️ CUDA NOT DETECTED - Troubleshooting:")
        logger_func("")
        
        if diagnostics['cuda_version'] is None:
            logger_func("  • Your PyTorch installation does NOT have CUDA support")
            logger_func("  • You are using a CPU-only version of PyTorch")
            logger_func("  • Solution: Reinstall PyTorch with CUDA support")
            logger_func("    Visit https://pytorch.org/get-started/locally/")
            logger_func("    Select your CUDA version and run the install command")
        else:
            logger_func("  • PyTorch has CUDA support but cannot detect GPU")
            logger_func("  • Possible causes:")
            logger_func("    - No NVIDIA GPU in system")
            logger_func("    - NVIDIA drivers not installed or outdated")
            logger_func("    - CUDA_VISIBLE_DEVICES is hiding GPUs")
            logger_func("    - Incompatible CUDA/driver version")
            logger_func("  • Verification steps:")
            logger_func("    1. Run 'nvidia-smi' to check GPU and driver")
            logger_func("    2. Check NVIDIA driver version matches CUDA requirements")
            logger_func("    3. Ensure CUDA_VISIBLE_DEVICES is not set to -1 or empty")
        logger_func("")
        logger_func("  Training will use CPU (slower but functional)")
        logger_func("=" * 60)


def get_device(preference='auto'):
    """
    Get PyTorch device based on preference.
    
    Args:
        preference: 'auto', 'force_gpu', or 'force_cpu'
    
    Returns:
        tuple: (torch.device, device_name, warning_message)
    """
    warning = None
    
    if preference == 'force_cpu':
        device = torch.device('cpu')
        device_name = 'CPU'
    elif preference == 'force_gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device('cpu')
            device_name = 'CPU'
            # Provide detailed warning
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
            if cuda_version is None:
                warning = "GPU requested but your PyTorch is CPU-only. Reinstall PyTorch with CUDA: https://pytorch.org"
            else:
                warning = "GPU requested but CUDA not available. Check drivers with nvidia-smi"
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device('cpu')
            device_name = 'CPU'
    
    return device, device_name, warning
