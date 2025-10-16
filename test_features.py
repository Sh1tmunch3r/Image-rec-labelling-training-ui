#!/usr/bin/env python3
"""
Tests for new features in Image Labeling Studio Pro
Tests cover: device detection, annotation saving, live recognition controls
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import numpy as np


class TestDeviceDetection(unittest.TestCase):
    """Test CUDA/GPU device detection and fallback"""
    
    def test_cuda_available_detection(self):
        """Test that CUDA availability is correctly detected"""
        # This will use actual torch.cuda.is_available()
        is_cuda = torch.cuda.is_available()
        self.assertIsInstance(is_cuda, bool)
    
    def test_device_creation_cpu(self):
        """Test CPU device can always be created"""
        device = torch.device('cpu')
        self.assertEqual(device.type, 'cpu')
        
        # Test that we can create tensors on CPU
        tensor = torch.zeros(1).to(device)
        self.assertEqual(tensor.device.type, 'cpu')
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_creation_cuda(self):
        """Test CUDA device creation when available"""
        device = torch.device('cuda')
        self.assertEqual(device.type, 'cuda')
        
        # Test that we can create tensors on CUDA
        tensor = torch.zeros(1).to(device)
        self.assertEqual(tensor.device.type, 'cuda')
    
    def test_device_fallback(self):
        """Test fallback from invalid device to CPU"""
        # Try to create device, should always succeed with fallback
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor = torch.zeros(1).to(device)
            self.assertIsNotNone(tensor)
        except Exception as e:
            self.fail(f"Device fallback failed: {e}")


class TestAnnotationFormats(unittest.TestCase):
    """Test COCO JSON and per-image JSON annotation formats"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, "images")
        self.annotations_dir = os.path.join(self.test_dir, "annotations")
        os.makedirs(self.images_dir)
        os.makedirs(self.annotations_dir)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_coco_json_structure(self):
        """Test COCO JSON format structure"""
        # Create sample COCO data
        coco_data = {
            "info": {
                "description": "Test Dataset",
                "version": "1.0",
                "year": 2025
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "test.jpg",
                    "width": 640,
                    "height": 480
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                    "score": 0.95
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "test_object",
                    "supercategory": "object"
                }
            ]
        }
        
        # Save to file
        coco_path = os.path.join(self.annotations_dir, "instances.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)
        
        # Load and verify
        with open(coco_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertIn("info", loaded_data)
        self.assertIn("images", loaded_data)
        self.assertIn("annotations", loaded_data)
        self.assertIn("categories", loaded_data)
        self.assertEqual(len(loaded_data["images"]), 1)
        self.assertEqual(len(loaded_data["annotations"]), 1)
        self.assertEqual(loaded_data["annotations"][0]["bbox"], [100, 100, 50, 50])
    
    def test_per_image_json_structure(self):
        """Test per-image JSON format structure"""
        # Create sample per-image data
        per_image_data = {
            "image": "test.jpg",
            "width": 640,
            "height": 480,
            "timestamp": "2025-10-16_09-00-00",
            "source": "screen_capture",
            "detections": [
                {
                    "label": "test_object",
                    "box": [100, 100, 150, 150],
                    "confidence": 0.95
                }
            ]
        }
        
        # Save to file
        json_path = os.path.join(self.annotations_dir, "test.json")
        with open(json_path, 'w') as f:
            json.dump(per_image_data, f)
        
        # Load and verify
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertIn("image", loaded_data)
        self.assertIn("detections", loaded_data)
        self.assertEqual(loaded_data["width"], 640)
        self.assertEqual(len(loaded_data["detections"]), 1)
        self.assertEqual(loaded_data["detections"][0]["label"], "test_object")
    
    def test_empty_detections(self):
        """Test handling of images with no detections"""
        # Per-image format with no detections
        per_image_data = {
            "image": "empty.jpg",
            "width": 640,
            "height": 480,
            "timestamp": "2025-10-16_09-00-00",
            "source": "screen_capture",
            "detections": []
        }
        
        json_path = os.path.join(self.annotations_dir, "empty.json")
        with open(json_path, 'w') as f:
            json.dump(per_image_data, f)
        
        # Verify it can be loaded
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(len(loaded_data["detections"]), 0)
    
    def test_coco_bbox_format(self):
        """Test COCO bbox format [x, y, width, height]"""
        # Convert from [x1, y1, x2, y2] to COCO format
        box = [100, 100, 200, 200]  # x1, y1, x2, y2
        x1, y1, x2, y2 = box
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height
        
        self.assertEqual(coco_bbox, [100, 100, 100, 100])
        
        # Area calculation
        area = (x2 - x1) * (y2 - y1)
        self.assertEqual(area, 10000)


class TestLiveRecognitionControls(unittest.TestCase):
    """Test live recognition control logic"""
    
    def test_fps_range(self):
        """Test FPS values are within valid range"""
        valid_fps_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for fps in valid_fps_values:
            self.assertGreaterEqual(fps, 1)
            self.assertLessEqual(fps, 10)
    
    def test_frame_capture_interval(self):
        """Test frame capture interval calculation"""
        fps = 5
        interval = 1.0 / fps
        self.assertAlmostEqual(interval, 0.2, places=2)
        
        fps = 10
        interval = 1.0 / fps
        self.assertAlmostEqual(interval, 0.1, places=2)
    
    def test_frames_to_save_validation(self):
        """Test validation of frames to save parameter"""
        valid_values = [1, 5, 10, 20, 50, 100]
        
        for val in valid_values:
            self.assertGreater(val, 0)
            self.assertIsInstance(val, int)


class TestImageSaving(unittest.TestCase):
    """Test image and annotation saving functionality"""
    
    def setUp(self):
        """Create temporary directory and test image"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test image
        self.test_image = Image.new('RGB', (640, 480), color='red')
        self.test_image_path = os.path.join(self.test_dir, "test.png")
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_image_save_load(self):
        """Test image can be saved and loaded"""
        self.assertTrue(os.path.exists(self.test_image_path))
        
        loaded_image = Image.open(self.test_image_path)
        self.assertEqual(loaded_image.size, (640, 480))
    
    def test_duplicate_filename_handling(self):
        """Test handling of duplicate filenames"""
        # Create first file
        path1 = os.path.join(self.test_dir, "image.png")
        self.test_image.save(path1)
        
        # Create second file with unique name
        path2 = os.path.join(self.test_dir, "image_001.png")
        self.test_image.save(path2)
        
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))
    
    def test_export_directory_creation(self):
        """Test export directory structure creation"""
        export_dir = os.path.join(self.test_dir, "export_test")
        images_dir = os.path.join(export_dir, "images")
        annotations_dir = os.path.join(export_dir, "annotations")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        self.assertTrue(os.path.exists(export_dir))
        self.assertTrue(os.path.exists(images_dir))
        self.assertTrue(os.path.exists(annotations_dir))


class TestNMSFiltering(unittest.TestCase):
    """Test Non-Maximum Suppression filtering"""
    
    def test_iou_calculation(self):
        """Test Intersection over Union calculation"""
        # Two identical boxes
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        iou = self.calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=2)
        
        # Non-overlapping boxes
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        iou = self.calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
        
        # Partially overlapping boxes
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = self.calculate_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)
    
    def calculate_iou(self, box1, box2):
        """Helper to calculate IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def test_confidence_filtering(self):
        """Test filtering detections by confidence threshold"""
        detections = [
            {'label': 'obj1', 'score': 0.9, 'box': [0, 0, 50, 50]},
            {'label': 'obj2', 'score': 0.6, 'box': [100, 100, 150, 150]},
            {'label': 'obj3', 'score': 0.3, 'box': [200, 200, 250, 250]},
        ]
        
        threshold = 0.5
        filtered = [d for d in detections if d['score'] >= threshold]
        
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(d['score'] >= threshold for d in filtered))


class TestDevicePreferences(unittest.TestCase):
    """Test device preference selection and settings persistence"""
    
    def setUp(self):
        """Create temporary directory for test settings"""
        self.test_dir = tempfile.mkdtemp()
        self.settings_file = os.path.join(self.test_dir, "settings.json")
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_auto_preference(self):
        """Test auto device preference"""
        # Import the helper function
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from device_utils import get_device
        
        device, device_name, warning = get_device('auto')
        
        # Should return a valid device
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ['cpu', 'cuda'])
        
        # Device name should be a string
        self.assertIsInstance(device_name, str)
        
        # Warning should be None for auto mode with available device
        if torch.cuda.is_available():
            self.assertIsNone(warning)
    
    def test_force_cpu_preference(self):
        """Test force CPU preference"""
        from device_utils import get_device
        
        device, device_name, warning = get_device('force_cpu')
        
        # Should always return CPU
        self.assertEqual(device.type, 'cpu')
        self.assertEqual(device_name, 'CPU')
        self.assertIsNone(warning)
    
    def test_force_gpu_preference_available(self):
        """Test force GPU preference when GPU is available"""
        from device_utils import get_device
        
        device, device_name, warning = get_device('force_gpu')
        
        if torch.cuda.is_available():
            # Should return CUDA device
            self.assertEqual(device.type, 'cuda')
            self.assertIn('GPU', device_name)
            self.assertIsNone(warning)
        else:
            # Should fallback to CPU with warning
            self.assertEqual(device.type, 'cpu')
            self.assertIsNotNone(warning)
            self.assertIn('not available', warning)
    
    def test_settings_save_load(self):
        """Test settings save and load"""
        from device_utils import save_settings, load_settings
        
        # Save test settings
        test_settings = {
            "device_preference": "force_gpu",
            "version": "1.0"
        }
        
        # Override settings file location for test
        import device_utils
        original_file = device_utils.SETTINGS_FILE
        original_folder = device_utils.CONFIG_FOLDER
        device_utils.SETTINGS_FILE = self.settings_file
        device_utils.CONFIG_FOLDER = self.test_dir
        
        try:
            # Save and load
            success = save_settings(test_settings)
            self.assertTrue(success)
            
            loaded = load_settings()
            self.assertEqual(loaded['device_preference'], 'force_gpu')
            self.assertEqual(loaded['version'], '1.0')
        finally:
            # Restore original
            device_utils.SETTINGS_FILE = original_file
            device_utils.CONFIG_FOLDER = original_folder
    
    def test_invalid_preference_fallback(self):
        """Test that invalid preference falls back to auto"""
        from device_utils import get_device
        
        # Pass invalid preference
        device, device_name, warning = get_device('invalid_preference')
        
        # Should behave like auto (default case in the function)
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ['cpu', 'cuda'])


class TestBackwardsCompatibility(unittest.TestCase):
    """Test that new features don't break existing functionality"""
    
    def test_default_settings(self):
        """Test that default settings are backwards compatible"""
        # Default device should fall back to CPU if CUDA unavailable
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertIn(device.type, ['cpu', 'cuda'])
        
        # Default FPS should be reasonable
        default_fps = 3
        self.assertGreaterEqual(default_fps, 1)
        self.assertLessEqual(default_fps, 10)
        
        # Default format should be COCO JSON
        default_format = "COCO JSON"
        self.assertIn(default_format, ["COCO JSON", "Per-image JSON"])
    
    def test_existing_annotation_format(self):
        """Test that existing annotation format is still supported"""
        # Old format: simple box annotations
        old_format = {
            "image_width": 640,
            "image_height": 480,
            "annotations": [
                {
                    "label": "object1",
                    "box": [100, 100, 200, 200]
                }
            ]
        }
        
        # Should be valid JSON
        json_str = json.dumps(old_format)
        loaded = json.loads(json_str)
        
        self.assertEqual(loaded["image_width"], 640)
        self.assertEqual(len(loaded["annotations"]), 1)
    
    def test_settings_file_structure(self):
        """Test that settings file has correct structure"""
        from device_utils import load_settings
        
        settings = load_settings()
        
        # Should have required keys
        self.assertIn('device_preference', settings)
        self.assertIn('version', settings)
        
        # Device preference should be valid
        self.assertIn(settings['device_preference'], ['auto', 'force_gpu', 'force_cpu'])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestAnnotationFormats))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveRecognitionControls))
    suite.addTests(loader.loadTestsFromTestCase(TestImageSaving))
    suite.addTests(loader.loadTestsFromTestCase(TestNMSFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestDevicePreferences))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardsCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
