"""
Device selection utilities for training
Separated from main app to allow testing without GUI dependencies
"""
import torch
import json
import os

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
            warning = "GPU requested but not available - falling back to CPU"
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device('cpu')
            device_name = 'CPU'
    
    return device, device_name, warning
