#!/usr/bin/env python3
"""
YOLOv5 Detection Module
Detects people, bags, suitcases, and other objects for surveillance.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import time

class YOLODetector:
    """YOLOv5 object detector for surveillance."""
    
    def __init__(self, model_size='s', device='auto', conf_threshold=0.7):
        """
        Initialize YOLOv5 detector.
        
        Args:
            model_size: 's', 'm', 'l', or 'x' for different model sizes
            device: 'auto', 'cpu', or 'cuda'
            conf_threshold: Confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load YOLOv5 model (using improved 'u' variant)
        model_name = f'yolov5{model_size}u.pt'
        self.model = YOLO(model_name)
        
        # Expanded target classes for comprehensive surveillance
        self.target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            15: 'cat',
            16: 'dog',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            31: 'bottle',
            32: 'wine glass',
            33: 'cup',
            39: 'banana',
            40: 'apple',
            41: 'sandwich',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        
        # Object categories for abandonment detection
        self.portable_objects = {24, 25, 26, 27, 28, 31, 32, 33, 39, 40, 41, 63, 64, 65, 66, 67, 73, 75, 76, 77, 78, 79}  # Objects that can be abandoned
        
        print(f"[INFO] YOLOv5{model_size}u detector initialized")
        print(f"[INFO] Target classes: {list(self.target_classes.values())}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, class_id, class_name
        """
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        # Run inference with aggressive NMS to eliminate overlapping boxes
        results = self.model(frame, conf=self.conf_threshold, iou=0.3, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for target classes only
                    if cls_id in self.target_classes:
                        x1, y1, x2, y2 = xyxy
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': self.target_classes[cls_id],
                            'is_person': cls_id == 0,
                            'is_portable': cls_id in self.portable_objects,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
        
        inference_time = time.time() - start_time
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()
        
        # Colors for different object types
        colors = {
            'person': (0, 255, 0),      # Green
            'backpack': (255, 0, 0),    # Blue
            'handbag': (255, 0, 0),     # Blue
            'suitcase': (0, 0, 255),    # Red
            'bottle': (255, 255, 0),    # Cyan
            'laptop': (255, 0, 255),    # Magenta
            'cell phone': (0, 255, 255) # Yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame
    
    def get_people_and_objects(self, detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate people and portable objects from detections.
        
        Args:
            detections: All detections
            
        Returns:
            Tuple of (people_detections, object_detections)
        """
        people = [det for det in detections if det['is_person']]
        objects = [det for det in detections if det['is_portable']]
        
        return people, objects
