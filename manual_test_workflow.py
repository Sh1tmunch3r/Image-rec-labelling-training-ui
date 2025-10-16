#!/usr/bin/env python3
"""
Manual test script to verify dataset export-to-training workflow
Run this to test the complete flow without the GUI
"""

import os
import json
import tempfile
import shutil
from PIL import Image
from dataset_utils import validate_dataset, register_dataset_as_project

def create_mock_export_dataset(export_dir):
    """Create a mock exported dataset for testing"""
    print(f"Creating mock dataset in {export_dir}...")
    
    images_dir = os.path.join(export_dir, "images")
    annotations_dir = os.path.join(export_dir, "annotations")
    os.makedirs(images_dir)
    os.makedirs(annotations_dir)
    
    # Create some test images and annotations
    for i in range(5):
        # Create image
        img = Image.new('RGB', (640, 480), color=(i*50, 100, 150))
        img_filename = f"capture_{i:03d}.png"
        img.save(os.path.join(images_dir, img_filename))
        
        # Create annotation (export format with 'detections' key)
        ann_data = {
            "image": img_filename,
            "width": 640,
            "height": 480,
            "timestamp": f"2025-10-16_14-{i:02d}-00",
            "source": "screen_capture",
            "detections": [
                {
                    "label": "button",
                    "box": [50 + i*20, 50 + i*20, 150 + i*20, 150 + i*20],
                    "confidence": 0.9 - i*0.1
                },
                {
                    "label": "icon",
                    "box": [200 + i*10, 200 + i*10, 250 + i*10, 250 + i*10],
                    "confidence": 0.85 - i*0.05
                }
            ]
        }
        
        ann_filename = f"capture_{i:03d}.json"
        with open(os.path.join(annotations_dir, ann_filename), 'w') as f:
            json.dump(ann_data, f, indent=2)
    
    print(f"  ✓ Created {5} images")
    print(f"  ✓ Created {5} annotation files")
    return export_dir

def test_workflow():
    """Test the complete export-to-training workflow"""
    print("\n" + "="*60)
    print("TESTING DATASET EXPORT-TO-TRAINING WORKFLOW")
    print("="*60 + "\n")
    
    # Create temporary directories
    test_dir = tempfile.mkdtemp(prefix="workflow_test_")
    export_dir = os.path.join(test_dir, "export_recognition_test")
    projects_dir = os.path.join(test_dir, "projects")
    os.makedirs(projects_dir)
    
    print(f"Test directory: {test_dir}\n")
    
    try:
        # Step 1: Create mock export dataset
        print("Step 1: Creating mock export dataset")
        print("-" * 60)
        create_mock_export_dataset(export_dir)
        print()
        
        # Step 2: Validate dataset
        print("Step 2: Validating exported dataset")
        print("-" * 60)
        is_valid, status = validate_dataset(export_dir)
        
        print(f"Validation result: {'✓ PASS' if is_valid else '✗ FAIL'}")
        print(f"  Images found: {status['image_count']}")
        print(f"  Annotated images: {status['valid_annotations']}")
        print(f"  Classes detected: {len(status['classes'])} - {sorted(status['classes'])}")
        
        if status['warnings']:
            print(f"  Warnings: {len(status['warnings'])}")
            for warning in status['warnings'][:3]:
                print(f"    ⚠️  {warning}")
        
        if status['errors']:
            print(f"  Errors: {len(status['errors'])}")
            for error in status['errors']:
                print(f"    ❌ {error}")
            print("\n❌ TEST FAILED: Dataset validation failed")
            return False
        
        print()
        
        # Step 3: Register as project
        print("Step 3: Registering dataset as training project")
        print("-" * 60)
        success, project_path, message = register_dataset_as_project(
            export_dir, 
            "test_workflow_project",
            projects_folder=projects_dir
        )
        
        if not success:
            print(f"❌ Registration failed: {message}")
            print("\n❌ TEST FAILED: Dataset registration failed")
            return False
        
        print("✓ Registration successful")
        print(f"  Project path: {project_path}")
        print(f"  {message.splitlines()[0]}")
        print()
        
        # Step 4: Verify project structure
        print("Step 4: Verifying project structure")
        print("-" * 60)
        checks = [
            ("images directory", os.path.join(project_path, "images")),
            ("annotations directory", os.path.join(project_path, "annotations")),
            ("classes.txt file", os.path.join(project_path, "classes.txt")),
        ]
        
        all_checks_passed = True
        for check_name, check_path in checks:
            exists = os.path.exists(check_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {check_name}: {check_path}")
            if not exists:
                all_checks_passed = False
        
        if not all_checks_passed:
            print("\n❌ TEST FAILED: Project structure incomplete")
            return False
        
        print()
        
        # Step 5: Verify classes.txt content
        print("Step 5: Verifying classes.txt content")
        print("-" * 60)
        classes_path = os.path.join(project_path, "classes.txt")
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        print(f"  Classes in classes.txt: {classes}")
        expected_classes = {"button", "icon"}
        actual_classes = set(classes)
        
        if expected_classes == actual_classes:
            print("  ✓ All expected classes present")
        else:
            print(f"  ✗ Class mismatch!")
            print(f"    Expected: {expected_classes}")
            print(f"    Actual: {actual_classes}")
            print("\n❌ TEST FAILED: Classes mismatch")
            return False
        
        print()
        
        # Step 6: Validate registered project
        print("Step 6: Validating registered project (ready for training)")
        print("-" * 60)
        is_valid, status = validate_dataset(project_path)
        
        if not is_valid:
            print("  ✗ Registered project validation failed")
            for error in status['errors']:
                print(f"    ❌ {error}")
            print("\n❌ TEST FAILED: Registered project not valid for training")
            return False
        
        print("  ✓ Project is valid and ready for training")
        print(f"    Images: {status['image_count']}")
        print(f"    Valid annotations: {status['valid_annotations']}")
        print(f"    Classes: {len(status['classes'])}")
        print()
        
        # Success!
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nWorkflow verification successful!")
        print("The export-to-training pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: Exception occurred")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)
        print("✓ Cleanup complete\n")

if __name__ == "__main__":
    success = test_workflow()
    exit(0 if success else 1)
