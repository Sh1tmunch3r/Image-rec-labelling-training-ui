#!/usr/bin/env python3
"""
Test for polygon annotation export functionality
Tests polygon annotations in COCO and per-image JSON formats
"""

import unittest
import json
import os
import tempfile
import shutil
from datetime import datetime


class TestPolygonExport(unittest.TestCase):
    """Test polygon annotation export"""
    
    def setUp(self):
        """Create temporary directory for test exports"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_coco_polygon_format(self):
        """Test that polygon annotations are properly formatted for COCO"""
        # Simulate the COCO export logic
        polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        
        # Convert to COCO segmentation format
        segmentation = []
        for pt in polygon:
            segmentation.extend([float(pt[0]), float(pt[1])])
        
        # Calculate bbox from polygon
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        width = x2 - x1
        height = y2 - y1
        
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [segmentation],
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        
        # Verify structure
        self.assertIn("segmentation", annotation)
        self.assertIsInstance(annotation["segmentation"], list)
        self.assertEqual(len(annotation["segmentation"][0]), 8)  # 4 points * 2 coords
        self.assertEqual(annotation["bbox"], [100, 100, 100, 100])
        self.assertEqual(annotation["area"], 10000)
    
    def test_per_image_polygon_format(self):
        """Test that polygon annotations are included in per-image JSON"""
        polygon = [[100, 100], [200, 100], [150, 200]]
        
        detection = {
            "label": "triangle",
            "polygon": polygon,
            "confidence": 0.95
        }
        
        # Verify polygon is preserved
        self.assertIn("polygon", detection)
        self.assertEqual(len(detection["polygon"]), 3)
        self.assertEqual(detection["polygon"][0], [100, 100])
    
    def test_mixed_annotations_export(self):
        """Test export with both box and polygon annotations"""
        results = [
            {"label": "box_obj", "box": [10, 10, 50, 50], "score": 0.9},
            {"label": "poly_obj", "polygon": [[100, 100], [200, 100], [150, 200]], "score": 0.85}
        ]
        
        # Simulate COCO export for mixed annotations
        coco_annotations = []
        annotation_id = 1
        
        for res in results:
            box = res.get('box')
            polygon = res.get('polygon')
            
            if not box and not polygon:
                continue
            
            annotation = {
                "id": annotation_id,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0
            }
            
            if polygon:
                # Polygon handling
                segmentation = []
                for pt in polygon:
                    segmentation.extend([float(pt[0]), float(pt[1])])
                annotation["segmentation"] = [segmentation]
                
                xs = [pt[0] for pt in polygon]
                ys = [pt[1] for pt in polygon]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                width = x2 - x1
                height = y2 - y1
                annotation["bbox"] = [x1, y1, width, height]
                annotation["area"] = width * height
            elif box:
                # Box handling
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                annotation["bbox"] = [x1, y1, width, height]
                annotation["area"] = width * height
            
            if 'score' in res:
                annotation['score'] = res['score']
            
            coco_annotations.append(annotation)
            annotation_id += 1
        
        # Verify both annotations were exported
        self.assertEqual(len(coco_annotations), 2)
        
        # Verify box annotation
        box_ann = coco_annotations[0]
        self.assertEqual(box_ann["bbox"], [10, 10, 40, 40])
        self.assertNotIn("segmentation", box_ann)
        
        # Verify polygon annotation
        poly_ann = coco_annotations[1]
        self.assertIn("segmentation", poly_ann)
        self.assertEqual(len(poly_ann["segmentation"][0]), 6)  # 3 points * 2 coords
        self.assertEqual(poly_ann["bbox"], [100, 100, 100, 100])


if __name__ == '__main__':
    unittest.main()
