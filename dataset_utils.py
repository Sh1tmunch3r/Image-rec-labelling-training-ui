"""
Dataset validation and registration utilities
Separated from main app to allow testing without GUI dependencies
"""
import os
import json
import shutil
from datetime import datetime

PROJECTS_FOLDER = "projects"


def validate_dataset(dataset_path):
    """
    Validate a dataset to check if it's ready for training.
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        tuple: (is_valid, status_dict) where status_dict contains diagnostic information
    """
    status = {
        "valid": False,
        "images_dir_exists": False,
        "annotations_dir_exists": False,
        "image_count": 0,
        "annotation_count": 0,
        "valid_annotations": 0,
        "classes": set(),
        "errors": [],
        "warnings": []
    }
    
    # Check directory structure
    images_dir = os.path.join(dataset_path, "images")
    annotations_dir = os.path.join(dataset_path, "annotations")
    
    status["images_dir_exists"] = os.path.exists(images_dir)
    status["annotations_dir_exists"] = os.path.exists(annotations_dir)
    
    if not status["images_dir_exists"]:
        status["errors"].append("Missing 'images' directory")
    
    if not status["annotations_dir_exists"]:
        status["errors"].append("Missing 'annotations' directory")
    
    if not status["images_dir_exists"] or not status["annotations_dir_exists"]:
        return False, status
    
    # Count images
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    status["image_count"] = len(images)
    
    if status["image_count"] == 0:
        status["errors"].append("No images found in dataset")
        return False, status
    
    # Validate annotations
    for img_file in images:
        ann_file = os.path.splitext(img_file)[0] + '.json'
        ann_path = os.path.join(annotations_dir, ann_file)
        
        if os.path.exists(ann_path):
            status["annotation_count"] += 1
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                # Support both formats (annotations or detections key)
                annotations = data.get('annotations', data.get('detections', []))
                
                if annotations:
                    status["valid_annotations"] += 1
                    for ann in annotations:
                        label = ann.get('label')
                        if label:
                            status["classes"].add(label)
                        if not ann.get('box'):
                            status["warnings"].append(f"{ann_file}: Missing box data")
                else:
                    status["warnings"].append(f"{ann_file}: No annotations/detections found")
                    
            except json.JSONDecodeError:
                status["errors"].append(f"{ann_file}: Invalid JSON format")
            except Exception as e:
                status["errors"].append(f"{ann_file}: {str(e)}")
    
    # Check if we have enough valid data
    if status["valid_annotations"] == 0:
        status["errors"].append("No valid annotated images found")
        return False, status
    
    if len(status["classes"]) == 0:
        status["errors"].append("No class labels found in annotations")
        return False, status
    
    # Dataset is valid if we have images, annotations, and no critical errors
    status["valid"] = (status["image_count"] > 0 and 
                      status["valid_annotations"] > 0 and 
                      len(status["classes"]) > 0)
    
    if status["annotation_count"] < status["image_count"]:
        missing = status["image_count"] - status["annotation_count"]
        status["warnings"].append(f"{missing} images without annotations")
    
    return status["valid"], status


def register_dataset_as_project(dataset_path, project_name=None, projects_folder=None):
    """
    Register an exported dataset as a project for training.
    
    Args:
        dataset_path: Path to the exported dataset
        project_name: Optional name for the project (auto-generated if None)
        projects_folder: Optional custom projects folder (uses default if None)
    
    Returns:
        tuple: (success, project_path, message)
    """
    if projects_folder is None:
        projects_folder = PROJECTS_FOLDER
    
    # Validate dataset first
    is_valid, status = validate_dataset(dataset_path)
    
    if not is_valid:
        error_msg = "Dataset validation failed:\n"
        for error in status["errors"]:
            error_msg += f"  • {error}\n"
        return False, None, error_msg
    
    # Generate project name if not provided
    if not project_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"exported_dataset_{timestamp}"
    
    # Create project in projects folder
    project_path = os.path.join(projects_folder, project_name)
    
    # Check if project already exists
    if os.path.exists(project_path):
        # Use existing project path
        pass
    else:
        # Create new project by copying dataset
        try:
            os.makedirs(project_path, exist_ok=True)
            
            # Copy images and annotations
            src_images = os.path.join(dataset_path, "images")
            dst_images = os.path.join(project_path, "images")
            src_annotations = os.path.join(dataset_path, "annotations")
            dst_annotations = os.path.join(project_path, "annotations")
            
            if os.path.exists(dst_images):
                shutil.rmtree(dst_images)
            if os.path.exists(dst_annotations):
                shutil.rmtree(dst_annotations)
                
            shutil.copytree(src_images, dst_images)
            shutil.copytree(src_annotations, dst_annotations)
            
            # Create classes.txt from detected classes
            classes_path = os.path.join(project_path, "classes.txt")
            with open(classes_path, 'w') as f:
                for class_name in sorted(status["classes"]):
                    f.write(f"{class_name}\n")
            
        except Exception as e:
            return False, None, f"Failed to create project: {str(e)}"
    
    success_msg = (f"Dataset registered successfully!\n"
                  f"Project: {project_name}\n"
                  f"Images: {status['image_count']}\n"
                  f"Valid annotations: {status['valid_annotations']}\n"
                  f"Classes: {len(status['classes'])}")
    
    if status["warnings"]:
        success_msg += f"\n\nWarnings:\n"
        for warning in status["warnings"][:5]:  # Show first 5 warnings
            success_msg += f"  • {warning}\n"
    
    return True, project_path, success_msg
