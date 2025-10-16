#!/usr/bin/env python3
"""
Tests for dataset registration and validation functionality
Tests cover: export-to-training flow, dataset validation, auto-selection
"""

import unittest
import json
import os
import tempfile
import shutil
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import validation functions from dataset_utils (no GUI dependencies)
from dataset_utils import validate_dataset, register_dataset_as_project


class TestDatasetValidation(unittest.TestCase):
    """Test dataset validation functionality"""
    
    def setUp(self):
        """Create temporary directory for test datasets"""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.test_dir, "test_dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        os.makedirs(self.images_dir)
        os.makedirs(self.annotations_dir)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def create_test_image(self, filename):
        """Helper to create a test image"""
        img = Image.new('RGB', (640, 480), color='blue')
        img_path = os.path.join(self.images_dir, filename)
        img.save(img_path)
        return img_path
    
    def create_test_annotation(self, filename, detections):
        """Helper to create a test annotation file"""
        data = {
            "image": filename,
            "width": 640,
            "height": 480,
            "detections": detections
        }
        ann_path = os.path.join(self.annotations_dir, filename.replace('.png', '.json'))
        with open(ann_path, 'w') as f:
            json.dump(data, f)
        return ann_path
    
    def test_valid_dataset(self):
        """Test validation of a valid dataset"""
        # Create test dataset
        self.create_test_image("test1.png")
        self.create_test_annotation("test1.png", [
            {"label": "cat", "box": [100, 100, 200, 200], "confidence": 0.9}
        ])
        
        self.create_test_image("test2.png")
        self.create_test_annotation("test2.png", [
            {"label": "dog", "box": [150, 150, 250, 250], "confidence": 0.85}
        ])
        
        # Import and test validation
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertTrue(is_valid)
        self.assertEqual(status["image_count"], 2)
        self.assertEqual(status["valid_annotations"], 2)
        self.assertEqual(len(status["classes"]), 2)
        self.assertIn("cat", status["classes"])
        self.assertIn("dog", status["classes"])
    
    def test_empty_dataset(self):
        """Test validation of empty dataset"""
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertFalse(is_valid)
        self.assertEqual(status["image_count"], 0)
        self.assertIn("No images found", str(status["errors"]))
    
    def test_polygon_annotations(self):
        """Test validation of polygon annotations"""
        # Create test dataset with polygon annotations
        self.create_test_image("test_poly.png")
        polygon_data = {
            "image": "test_poly.png",
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "label": "triangle",
                    "type": "polygon",
                    "polygon": [[100, 100], [200, 100], [150, 200]]
                }
            ]
        }
        ann_path = os.path.join(self.annotations_dir, "test_poly.json")
        with open(ann_path, 'w') as f:
            json.dump(polygon_data, f)
        
        # Validate dataset
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertTrue(is_valid, "Polygon annotations should be valid")
        self.assertEqual(status["valid_annotations"], 1)
        self.assertIn("triangle", status["classes"])
        # Should not have warning about missing box for polygon annotations
        box_warnings = [w for w in status["warnings"] if "Missing box" in w and "polygon" not in w.lower()]
        self.assertEqual(len(box_warnings), 0, "Should not warn about missing box for polygon annotations")
    
    def test_mixed_box_and_polygon_annotations(self):
        """Test validation of dataset with both box and polygon annotations"""
        # Image with box annotation
        self.create_test_image("test_box.png")
        self.create_test_annotation("test_box.png", [
            {"label": "cat", "box": [100, 100, 200, 200]}
        ])
        
        # Image with polygon annotation
        self.create_test_image("test_poly.png")
        polygon_data = {
            "image": "test_poly.png",
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "label": "dog",
                    "type": "polygon",
                    "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]
                }
            ]
        }
        ann_path = os.path.join(self.annotations_dir, "test_poly.json")
        with open(ann_path, 'w') as f:
            json.dump(polygon_data, f)
        
        # Validate dataset
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertTrue(is_valid, "Mixed annotations should be valid")
        self.assertEqual(status["image_count"], 2)
        self.assertEqual(status["valid_annotations"], 2)
        self.assertEqual(len(status["classes"]), 2)
        self.assertIn("cat", status["classes"])
        self.assertIn("dog", status["classes"])
    
    def test_missing_annotations(self):
        """Test dataset with images but no annotations"""
        self.create_test_image("test1.png")
        self.create_test_image("test2.png")
        # No annotations created
        
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertFalse(is_valid)
        self.assertEqual(status["image_count"], 2)
        self.assertEqual(status["valid_annotations"], 0)
        self.assertIn("No valid annotated images", str(status["errors"]))
    
    def test_annotations_format_support(self):
        """Test both 'detections' and 'annotations' key formats"""
        # Test with 'detections' key (export format)
        self.create_test_image("test1.png")
        data1 = {
            "image": "test1.png",
            "detections": [{"label": "obj1", "box": [0, 0, 50, 50]}]
        }
        ann_path1 = os.path.join(self.annotations_dir, "test1.json")
        with open(ann_path1, 'w') as f:
            json.dump(data1, f)
        
        # Test with 'annotations' key (project format)
        self.create_test_image("test2.png")
        data2 = {
            "image_width": 640,
            "image_height": 480,
            "annotations": [{"label": "obj2", "box": [100, 100, 150, 150]}]
        }
        ann_path2 = os.path.join(self.annotations_dir, "test2.json")
        with open(ann_path2, 'w') as f:
            json.dump(data2, f)
        
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertTrue(is_valid)
        self.assertEqual(status["valid_annotations"], 2)
    
    def test_empty_annotations(self):
        """Test images with empty annotation arrays"""
        self.create_test_image("test1.png")
        self.create_test_annotation("test1.png", [])  # Empty detections
        
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertFalse(is_valid)
        self.assertIn("No valid annotated images", str(status["errors"]))
    
    def test_invalid_json(self):
        """Test handling of invalid JSON files"""
        self.create_test_image("test1.png")
        
        # Create invalid JSON
        ann_path = os.path.join(self.annotations_dir, "test1.json")
        with open(ann_path, 'w') as f:
            f.write("{invalid json content")
        
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid JSON" in str(err) for err in status["errors"]))
    
    def test_mixed_valid_invalid(self):
        """Test dataset with mix of valid and invalid annotations"""
        # Valid annotation
        self.create_test_image("test1.png")
        self.create_test_annotation("test1.png", [
            {"label": "valid", "box": [0, 0, 50, 50]}
        ])
        
        # Image without annotation
        self.create_test_image("test2.png")
        
        # Image with empty annotation
        self.create_test_image("test3.png")
        self.create_test_annotation("test3.png", [])
        
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(self.dataset_dir)
        
        # Should be valid because we have at least 1 valid annotation
        self.assertTrue(is_valid)
        self.assertEqual(status["valid_annotations"], 1)
        self.assertEqual(status["image_count"], 3)
        self.assertGreater(len(status["warnings"]), 0)


class TestDatasetRegistration(unittest.TestCase):
    """Test dataset registration as project"""
    
    def setUp(self):
        """Create temporary directories"""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.test_dir, "export_dataset")
        self.projects_dir = os.path.join(self.test_dir, "projects")
        os.makedirs(self.dataset_dir)
        os.makedirs(self.projects_dir)
        
        # Create valid dataset
        images_dir = os.path.join(self.dataset_dir, "images")
        annotations_dir = os.path.join(self.dataset_dir, "annotations")
        os.makedirs(images_dir)
        os.makedirs(annotations_dir)
        
        # Add test data
        img = Image.new('RGB', (640, 480), color='green')
        img.save(os.path.join(images_dir, "test.png"))
        
        ann_data = {
            "image": "test.png",
            "detections": [
                {"label": "object1", "box": [50, 50, 100, 100]},
                {"label": "object2", "box": [200, 200, 300, 300]}
            ]
        }
        with open(os.path.join(annotations_dir, "test.json"), 'w') as f:
            json.dump(ann_data, f)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_register_valid_dataset(self):
        """Test registering a valid dataset"""
        from dataset_utils import register_dataset_as_project, PROJECTS_FOLDER
        
        # Temporarily override PROJECTS_FOLDER
        import dataset_utils as du
        original_folder = du.PROJECTS_FOLDER
        du.PROJECTS_FOLDER = self.projects_dir
        
        try:
            success, project_path, message = register_dataset_as_project(
                self.dataset_dir, "test_project"
            )
            
            self.assertTrue(success)
            self.assertIsNotNone(project_path)
            self.assertIn("registered successfully", message)
            
            # Check project structure
            self.assertTrue(os.path.exists(project_path))
            self.assertTrue(os.path.exists(os.path.join(project_path, "images")))
            self.assertTrue(os.path.exists(os.path.join(project_path, "annotations")))
            self.assertTrue(os.path.exists(os.path.join(project_path, "classes.txt")))
            
            # Check classes.txt
            with open(os.path.join(project_path, "classes.txt"), 'r') as f:
                classes = [line.strip() for line in f]
            self.assertIn("object1", classes)
            self.assertIn("object2", classes)
            
        finally:
            du.PROJECTS_FOLDER = original_folder
    
    def test_register_invalid_dataset(self):
        """Test registering an invalid dataset"""
        from dataset_utils import register_dataset_as_project
        
        # Create invalid dataset (no annotations)
        invalid_dir = os.path.join(self.test_dir, "invalid")
        os.makedirs(os.path.join(invalid_dir, "images"))
        os.makedirs(os.path.join(invalid_dir, "annotations"))
        
        success, project_path, message = register_dataset_as_project(
            invalid_dir, "invalid_project"
        )
        
        self.assertFalse(success)
        self.assertIsNone(project_path)
        self.assertIn("validation failed", message.lower())
    
    def test_auto_generated_project_name(self):
        """Test auto-generated project name"""
        from dataset_utils import register_dataset_as_project
        import dataset_utils as du
        
        original_folder = du.PROJECTS_FOLDER
        du.PROJECTS_FOLDER = self.projects_dir
        
        try:
            success, project_path, message = register_dataset_as_project(
                self.dataset_dir, None  # No project name
            )
            
            self.assertTrue(success)
            self.assertIsNotNone(project_path)
            
            # Check name contains timestamp
            project_name = os.path.basename(project_path)
            self.assertTrue(project_name.startswith("exported_dataset_"))
            
        finally:
            du.PROJECTS_FOLDER = original_folder


class TestAnnotationDatasetCompatibility(unittest.TestCase):
    """Test AnnotationDataset supports both export and project formats"""
    
    def setUp(self):
        """Create temporary dataset"""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.test_dir, "dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        os.makedirs(self.images_dir)
        os.makedirs(self.annotations_dir)
        
        # Create classes
        self.classes_path = os.path.join(self.dataset_dir, "classes.txt")
        with open(self.classes_path, 'w') as f:
            f.write("cat\ndog\n")
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_detections_key_format(self):
        """Test loading annotations with 'detections' key via validation"""
        # Create image
        img = Image.new('RGB', (640, 480), color='red')
        img.save(os.path.join(self.images_dir, "test.png"))
        
        # Create annotation with 'detections' key (export format)
        ann_data = {
            "image": "test.png",
            "detections": [
                {"label": "cat", "box": [10, 10, 50, 50]}
            ]
        }
        with open(os.path.join(self.annotations_dir, "test.json"), 'w') as f:
            json.dump(ann_data, f)
        
        # Validate using our utility
        is_valid, status = validate_dataset(self.dataset_dir)
        
        # Should recognize the 'detections' format
        self.assertTrue(is_valid)
        self.assertEqual(status['valid_annotations'], 1)
        self.assertIn("cat", status['classes'])
    
    def test_annotations_key_format(self):
        """Test loading annotations with 'annotations' key via validation"""
        # Create image
        img = Image.new('RGB', (640, 480), color='blue')
        img.save(os.path.join(self.images_dir, "test.png"))
        
        # Create annotation with 'annotations' key (project format)
        ann_data = {
            "image_width": 640,
            "image_height": 480,
            "annotations": [
                {"label": "dog", "box": [100, 100, 200, 200]}
            ]
        }
        with open(os.path.join(self.annotations_dir, "test.json"), 'w') as f:
            json.dump(ann_data, f)
        
        # Validate using our utility
        is_valid, status = validate_dataset(self.dataset_dir)
        
        # Should recognize the 'annotations' format  
        self.assertTrue(is_valid)
        self.assertEqual(status['valid_annotations'], 1)
        self.assertIn("dog", status['classes'])


class TestExportToTrainingFlow(unittest.TestCase):
    """Test complete export-to-training flow"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir)
    
    def test_complete_flow(self):
        """Test export -> validate -> register -> train flow"""
        # 1. Create export dataset
        export_dir = os.path.join(self.test_dir, "export")
        images_dir = os.path.join(export_dir, "images")
        annotations_dir = os.path.join(export_dir, "annotations")
        os.makedirs(images_dir)
        os.makedirs(annotations_dir)
        
        # Add test images and annotations
        for i in range(5):
            img = Image.new('RGB', (640, 480), color='red')
            img.save(os.path.join(images_dir, f"img_{i}.png"))
            
            ann_data = {
                "image": f"img_{i}.png",
                "detections": [
                    {"label": "obj1", "box": [i*10, i*10, i*10+50, i*10+50]},
                    {"label": "obj2", "box": [100+i*10, 100+i*10, 150+i*10, 150+i*10]}
                ]
            }
            with open(os.path.join(annotations_dir, f"img_{i}.json"), 'w') as f:
                json.dump(ann_data, f)
        
        # 2. Validate dataset
        from dataset_utils import validate_dataset
        is_valid, status = validate_dataset(export_dir)
        
        self.assertTrue(is_valid)
        self.assertEqual(status["image_count"], 5)
        self.assertEqual(status["valid_annotations"], 5)
        self.assertEqual(len(status["classes"]), 2)
        
        # 3. Register as project
        from dataset_utils import register_dataset_as_project
        import dataset_utils as du
        
        projects_dir = os.path.join(self.test_dir, "projects")
        os.makedirs(projects_dir)
        original_folder = du.PROJECTS_FOLDER
        du.PROJECTS_FOLDER = projects_dir
        
        try:
            success, project_path, message = register_dataset_as_project(
                export_dir, "test_training_project"
            )
            
            self.assertTrue(success)
            self.assertIn("registered successfully", message)
            
            # 4. Verify project is ready for training
            is_valid, status = validate_dataset(project_path)
            self.assertTrue(is_valid)
            
            # 5. Verify project structure is correct for training
            # Check that all required files exist
            self.assertTrue(os.path.exists(os.path.join(project_path, "images")))
            self.assertTrue(os.path.exists(os.path.join(project_path, "annotations")))
            self.assertTrue(os.path.exists(os.path.join(project_path, "classes.txt")))
            
            # Check classes.txt content
            with open(os.path.join(project_path, "classes.txt"), 'r') as f:
                classes = [line.strip() for line in f]
            self.assertEqual(len(classes), 2)
            self.assertIn("obj1", classes)
            self.assertIn("obj2", classes)
            
        finally:
            du.PROJECTS_FOLDER = original_folder


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestAnnotationDatasetCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestExportToTrainingFlow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
