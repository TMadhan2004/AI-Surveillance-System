#!/usr/bin/env python3
"""
Multi-Object Tracking Module
Uses DeepSORT for robust tracking of people and objects.
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Tuple
import time
from collections import defaultdict, deque

class SurveillanceTracker:
    """Multi-object tracker for surveillance system."""
    
    def __init__(self, max_age=15, n_init=2):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Number of consecutive detections before track is confirmed
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.3,
            nn_budget=50,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Track history for analysis
        self.track_history = defaultdict(lambda: deque(maxlen=100))
        self.track_speeds = defaultdict(list)
        self.track_directions = defaultdict(list)
        
        # Association tracking for abandonment detection
        self.object_associations = {}  # object_track_id -> person_track_id
        self.association_history = defaultdict(list)
        
        print("[INFO] DeepSORT tracker initialized")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame for embedding extraction
            
        Returns:
            List of tracked objects with track IDs
        """
        if not detections:
            # Update tracker with empty detections to age out tracks
            tracks = self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert detections to DeepSORT format
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # DeepSORT expects [[x1, y1, w, h], confidence, class_name]
            det_list.append([[x1, y1, x2-x1, y2-y1], confidence, class_name])
        
        # Update tracker
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        # Convert tracks back to our format
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Get detection info
            det_class = track.get_det_class() if hasattr(track, 'get_det_class') else 'unknown'
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            
            center = [(x1 + x2) // 2, (y1 + y2) // 2]
            
            tracked_obj = {
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'center': center,
                'class_name': det_class,
                'confidence': confidence,
                'is_person': det_class == 'person',
                'is_portable': det_class in ['backpack', 'handbag', 'suitcase', 'bottle', 'laptop', 'cell phone'],
                'area': (x2 - x1) * (y2 - y1)
            }
            
            # Update track history
            self.track_history[track_id].append({
                'center': center,
                'timestamp': time.time(),
                'bbox': [x1, y1, x2, y2]
            })
            
            # Calculate speed and direction
            self._update_kinematics(track_id)
            
            tracked_objects.append(tracked_obj)
        
        # Update object-person associations
        self._update_associations(tracked_objects)
        
        return tracked_objects
    
    def _update_kinematics(self, track_id: int):
        """Update speed and direction for a track."""
        history = self.track_history[track_id]
        
        if len(history) < 2:
            return
        
        # Calculate speed (pixels per second)
        current = history[-1]
        previous = history[-2]
        
        dt = current['timestamp'] - previous['timestamp']
        if dt > 0:
            dx = current['center'][0] - previous['center'][0]
            dy = current['center'][1] - previous['center'][1]
            
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / dt
            
            self.track_speeds[track_id].append(speed)
            if len(self.track_speeds[track_id]) > 30:
                self.track_speeds[track_id].pop(0)
            
            # Direction (angle in radians)
            if distance > 1:  # Avoid noise
                direction = np.arctan2(dy, dx)
                self.track_directions[track_id].append(direction)
                if len(self.track_directions[track_id]) > 30:
                    self.track_directions[track_id].pop(0)
    
    def _update_associations(self, tracked_objects: List[Dict]):
        """Update object-person associations for abandonment detection."""
        people = [obj for obj in tracked_objects if obj['is_person']]
        objects = [obj for obj in tracked_objects if obj['is_portable']]
        
        # Associate objects with nearest person
        for obj in objects:
            obj_center = obj['center']
            min_distance = float('inf')
            closest_person_id = None
            
            for person in people:
                person_center = person['center']
                distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                                 (obj_center[1] - person_center[1])**2)
                
                if distance < min_distance and distance < 150:  # Max association distance
                    min_distance = distance
                    closest_person_id = person['track_id']
            
            # Update association
            obj_id = obj['track_id']
            if closest_person_id:
                self.object_associations[obj_id] = closest_person_id
                self.association_history[obj_id].append({
                    'person_id': closest_person_id,
                    'distance': min_distance,
                    'timestamp': time.time()
                })
            
            # Keep only recent associations
            if obj_id in self.association_history:
                recent_time = time.time() - 30  # Keep 30 seconds
                self.association_history[obj_id] = [
                    assoc for assoc in self.association_history[obj_id]
                    if assoc['timestamp'] > recent_time
                ]
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """Get movement statistics for a track."""
        if track_id not in self.track_history:
            return {}
        
        history = self.track_history[track_id]
        speeds = self.track_speeds.get(track_id, [])
        directions = self.track_directions.get(track_id, [])
        
        stats = {
            'track_length': len(history),
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'speed_variance': np.var(speeds) if len(speeds) > 1 else 0,
            'direction_changes': 0,
            'is_stationary': False,
            'is_loitering': False
        }
        
        # Direction change analysis
        if len(directions) > 2:
            direction_diffs = np.diff(directions)
            # Handle angle wrapping
            direction_diffs = np.where(direction_diffs > np.pi, direction_diffs - 2*np.pi, direction_diffs)
            direction_diffs = np.where(direction_diffs < -np.pi, direction_diffs + 2*np.pi, direction_diffs)
            
            # Count significant direction changes (> 45 degrees)
            significant_changes = np.abs(direction_diffs) > np.pi/4
            stats['direction_changes'] = np.sum(significant_changes)
        
        # Stationary detection
        if stats['avg_speed'] < 5:  # pixels per second
            stats['is_stationary'] = True
        
        # Loitering detection (stationary for extended period)
        if len(history) > 150 and stats['avg_speed'] < 3:  # ~5 seconds at 30fps
            stats['is_loitering'] = True
        
        return stats
    
    def get_object_owner(self, object_track_id: int) -> int:
        """Get the person ID associated with an object."""
        return self.object_associations.get(object_track_id, None)
    
    def is_object_abandoned(self, object_track_id: int, current_people_ids: List[int]) -> bool:
        """Check if an object has been abandoned."""
        if object_track_id not in self.object_associations:
            return False
        
        owner_id = self.object_associations[object_track_id]
        
        # Check if owner is still present
        if owner_id not in current_people_ids:
            # Check how long the object has been without its owner
            history = self.association_history.get(object_track_id, [])
            if history:
                last_association = max(history, key=lambda x: x['timestamp'])
                time_since_owner = time.time() - last_association['timestamp']
                
                # Object is abandoned if owner has been gone for > 10 seconds
                # and object is stationary
                obj_stats = self.get_track_statistics(object_track_id)
                if time_since_owner > 10 and obj_stats.get('is_stationary', False):
                    return True
        
        return False
    
    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """Draw tracking information on frame."""
        vis_frame = frame.copy()
        
        # Colors for different track types
        colors = {
            'person': (0, 255, 0),      # Green
            'backpack': (255, 0, 0),    # Blue
            'handbag': (255, 0, 0),     # Blue
            'suitcase': (0, 0, 255),    # Red
            'bottle': (255, 255, 0),    # Cyan
            'laptop': (255, 0, 255),    # Magenta
            'cell phone': (0, 255, 255) # Yellow
        }
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            class_name = obj['class_name']
            
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{track_id} {class_name}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw track history
            if track_id in self.track_history:
                history = list(self.track_history[track_id])
                if len(history) > 1:
                    points = [h['center'] for h in history[-20:]]  # Last 20 points
                    for i in range(1, len(points)):
                        cv2.line(vis_frame, tuple(points[i-1]), tuple(points[i]), color, 2)
            
            # Draw associations for objects
            if obj['is_portable'] and track_id in self.object_associations:
                owner_id = self.object_associations[track_id]
                cv2.putText(vis_frame, f"Owner: {owner_id}", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame
