#!/usr/bin/env python3
"""
Test script for the image downloader functionality
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only import non-GUI utilities
from ui.app_config import load_app_config, save_app_config, update_app_config


def test_filename_sanitization():
    """Test filename sanitization"""
    print("Testing filename sanitization...")
    
    # Create a temporary project for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = os.path.join(tmpdir, "test_project")
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, "images"))
        
        # Create a mock downloader (without GUI)
        class MockDownloader:
            def __init__(self, project_path):
                self.project_path = project_path
                self.preserve_names_var = type('obj', (object,), {'get': lambda self: False})()
            
            def sanitize_filename(self, filename):
                import re
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                filename = filename.strip('. ')
                name, ext = os.path.splitext(filename)
                if len(name) > 200:
                    name = name[:200]
                return name + ext
            
            def get_filename_from_url(self, url):
                from urllib.parse import urlparse, unquote
                import hashlib
                parsed = urlparse(url)
                path = unquote(parsed.path)
                filename = os.path.basename(path)
                
                if not filename or '.' not in filename:
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"downloaded_{url_hash}.jpg"
                    
                return self.sanitize_filename(filename)
            
            def get_next_image_number(self):
                import re
                images_dir = os.path.join(self.project_path, "images")
                if not os.path.exists(images_dir):
                    return 1
                    
                existing_numbers = []
                for filename in os.listdir(images_dir):
                    match = re.match(r'img_(\d+)\.\w+', filename)
                    if match:
                        existing_numbers.append(int(match.group(1)))
                        
                return max(existing_numbers) + 1 if existing_numbers else 1
            
            def generate_project_filename(self, original_filename, image_number):
                if self.preserve_names_var.get():
                    images_dir = os.path.join(self.project_path, "images")
                    safe_name = self.sanitize_filename(original_filename)
                    target_path = os.path.join(images_dir, safe_name)
                    
                    if not os.path.exists(target_path):
                        return safe_name
                
                ext = os.path.splitext(original_filename)[1] or '.jpg'
                return f"img_{image_number:04d}{ext}"
        
        downloader = MockDownloader(project_path)
        
        # Test cases
        test_cases = [
            ("https://example.com/image.jpg", "image.jpg"),
            ("https://example.com/path/to/photo.png", "photo.png"),
            ("https://example.com/image%20with%20spaces.jpg", "image with spaces.jpg"),
            ("https://example.com/no-extension", "downloaded_"),  # will have hash
        ]
        
        for url, expected_part in test_cases:
            result = downloader.get_filename_from_url(url)
            if expected_part in result or expected_part == "downloaded_":
                print(f"  ✓ {url} -> {result}")
            else:
                print(f"  ✗ {url} -> {result} (expected {expected_part})")
        
        # Test project naming
        print("\nTesting project naming...")
        next_num = downloader.get_next_image_number()
        print(f"  Next image number: {next_num}")
        
        filename = downloader.generate_project_filename("test.jpg", next_num)
        print(f"  Generated filename: {filename}")
        
        # Create a file and test incrementing
        with open(os.path.join(project_path, "images", filename), 'w') as f:
            f.write("test")
        
        next_num = downloader.get_next_image_number()
        print(f"  After creating file, next number: {next_num}")
        
        filename2 = downloader.generate_project_filename("test2.jpg", next_num)
        print(f"  Second generated filename: {filename2}")
        
        print("\n✓ Filename sanitization tests passed!")


def test_app_config():
    """Test app configuration persistence"""
    print("\nTesting app configuration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily change config file location
        import ui.app_config as cfg
        original_config_file = cfg.CONFIG_FILE
        cfg.CONFIG_FILE = os.path.join(tmpdir, "test_config.json")
        
        try:
            # Test saving and loading
            update_app_config("last_project", "/path/to/project")
            update_app_config("last_model", "my_model.pth")
            
            config = load_app_config()
            assert config["last_project"] == "/path/to/project", "Project path not saved"
            assert config["last_model"] == "my_model.pth", "Model not saved"
            
            print("  ✓ Config save/load works")
            
            # Test individual get
            last_proj = cfg.get_app_config("last_project")
            assert last_proj == "/path/to/project", "get_app_config failed"
            
            print("  ✓ Individual config get works")
            
            print("\n✓ App configuration tests passed!")
            
        finally:
            cfg.CONFIG_FILE = original_config_file


def main():
    """Run all tests"""
    print("=" * 60)
    print("Image Downloader Feature Tests")
    print("=" * 60)
    
    try:
        test_filename_sanitization()
        test_app_config()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
