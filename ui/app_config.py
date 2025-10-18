"""
Application configuration management
Persists user preferences like last-selected project and model
"""

import json
import os
from typing import Dict, Any, Optional

CONFIG_FILE = "config/app_config.json"


def load_app_config() -> Dict[str, Any]:
    """
    Load application configuration from JSON file.
    
    Returns:
        Dictionary with configuration values
    """
    if not os.path.exists(CONFIG_FILE):
        return {
            "last_project": None,
            "last_model": None,
            "window_geometry": None,
            "last_download_folder": None
        }
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading app config: {e}")
        return {
            "last_project": None,
            "last_model": None,
            "window_geometry": None,
            "last_download_folder": None
        }


def save_app_config(config: Dict[str, Any]) -> None:
    """
    Save application configuration to JSON file.
    
    Args:
        config: Dictionary with configuration values
    """
    # Ensure config directory exists
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving app config: {e}")


def update_app_config(key: str, value: Any) -> None:
    """
    Update a single configuration value.
    
    Args:
        key: Configuration key
        value: New value
    """
    config = load_app_config()
    config[key] = value
    save_app_config(config)


def get_app_config(key: str, default: Any = None) -> Any:
    """
    Get a single configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_app_config()
    return config.get(key, default)
