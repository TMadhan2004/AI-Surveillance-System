#!/usr/bin/env python3
"""
Analytics Module for Object Abandonment and Unusual Movement Detection
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class Alert:
    """Alert data structure."""
    alert_type: str
    track_id: int
    confidence: float
    timestamp: float
    bbox: List[int]
    description: str
    metadata: Dict

class AbandonmentDetector:
    """Detects abandoned objects using tracking data."""
    
    def __init__(self, 
                 min_stationary_time: float = 10.0,
                 min_owner_distance: float = 200.0,
                 min_object_confirm_time: float = 3.0):
        """
        Initialize abandonment detector.
        
        Args:
            min_stationary_time: Minimum time object must be stationary (seconds)
            min_owner_distance: Minimum distance from owner to trigger alert (pixels)
            min_object_confirm_time: Minimum time object must exist before arming
        """
        self.min_stationary_time = min_stationary_time
        self.min_owner_distance = min_owner_distance
        self.min_object_confirm_time = min_object_confirm_time
        
        # Track object states
        self.object_states = {}
        self.abandonment_candidates = {}
        
    def update(self, tracked_objects: List[Dict], tracker) -> List[Alert]:
        """
        Update abandonment detection with new tracking data.
        
        Args:
            tracked_objects: Current tracked objects
            tracker: Tracker instance for getting associations
            
        Returns:
            List of abandonment alerts
        """
        alerts = []
        current_time = time.time()
        
        # Get current people and objects
        people = [obj for obj in tracked_objects if obj['is_person']]
        objects = [obj for obj in tracked_objects if obj['is_portable']]
        
        people_ids = [p['track_id'] for p in people]
        
        for obj in objects:
            obj_id = obj['track_id']
            
            # Initialize object state if new
            if obj_id not in self.object_states:
                self.object_states[obj_id] = {
                    'first_seen': current_time,
                    'last_movement': current_time,
                    'stationary_start': None,
                    'armed': False
                }
            
            state = self.object_states[obj_id]
            
            # Check if object has existed long enough to be armed
            if not state['armed']:
                if current_time - state['first_seen'] >= self.min_object_confirm_time:
                    state['armed'] = True
                continue
            
            # Get object movement statistics
            stats = tracker.get_track_statistics(obj_id)
            is_stationary = stats.get('is_stationary', False)
            
            # Update stationary tracking
            if is_stationary:
                if state['stationary_start'] is None:
                    state['stationary_start'] = current_time
            else:
                state['stationary_start'] = None
                state['last_movement'] = current_time
            
            # Check for abandonment
            if (state['stationary_start'] is not None and 
                current_time - state['stationary_start'] >= self.min_stationary_time):
                
                # Check if owner is present
                owner_id = tracker.get_object_owner(obj_id)
                if owner_id and owner_id not in people_ids:
                    # Object is abandoned
                    if obj_id not in self.abandonment_candidates:
                        alert = Alert(
                            alert_type="object_abandonment",
                            track_id=obj_id,
                            confidence=0.8,
                            timestamp=current_time,
                            bbox=obj['bbox'],
                            description=f"Abandoned {obj['class_name']} detected",
                            metadata={
                                'object_class': obj['class_name'],
                                'owner_id': owner_id,
                                'stationary_duration': current_time - state['stationary_start'],
                                'location': obj['center']
                            }
                        )
                        alerts.append(alert)
                        self.abandonment_candidates[obj_id] = current_time
        
        return alerts

class FightingDetector:
    """Detects fighting behavior using movement analysis."""
    
    def __init__(self, 
                 aggression_threshold: float = 150.0,
                 proximity_threshold: float = 100.0,
                 rapid_movement_threshold: float = 80.0):
        """
        Initialize fighting detector.
        
        Args:
            aggression_threshold: Speed threshold for aggressive movement
            proximity_threshold: Distance threshold for close interaction
            rapid_movement_threshold: Threshold for rapid back-and-forth movement
        """
        self.aggression_threshold = aggression_threshold
        self.proximity_threshold = proximity_threshold
        self.rapid_movement_threshold = rapid_movement_threshold
        self.interaction_history = defaultdict(list)
        self.alert_cooldowns = defaultdict(float)
    
    def detect_fighting(self, people: List[Dict], tracker, current_time: float) -> List[Alert]:
        """Detect potential fighting between people."""
        alerts = []
        
        # Check all pairs of people
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                person1, person2 = people[i], people[j]
                
                # Calculate distance between people
                p1_center = person1['center']
                p2_center = person2['center']
                distance = np.sqrt((p1_center[0] - p2_center[0])**2 + (p1_center[1] - p2_center[1])**2)
                
                # Check if people are in close proximity
                if distance < self.proximity_threshold:
                    # Get movement statistics for both people
                    stats1 = tracker.get_track_statistics(person1['track_id'])
                    stats2 = tracker.get_track_statistics(person2['track_id'])
                    
                    speed1 = stats1.get('max_speed', 0)
                    speed2 = stats2.get('max_speed', 0)
                    
                    # Check for aggressive movement patterns
                    if (speed1 > self.aggression_threshold or speed2 > self.aggression_threshold):
                        # Check direction changes (erratic movement)
                        changes1 = stats1.get('direction_changes', 0)
                        changes2 = stats2.get('direction_changes', 0)
                        
                        if changes1 > 8 or changes2 > 8:  # High erratic movement
                            pair_key = f"{min(person1['track_id'], person2['track_id'])}_{max(person1['track_id'], person2['track_id'])}"
                            
                            if current_time - self.alert_cooldowns[pair_key] > 15.0:
                                alert = Alert(
                                    alert_type="fighting_detected",
                                    track_id=person1['track_id'],
                                    confidence=0.85,
                                    timestamp=current_time,
                                    bbox=person1['bbox'],
                                    description=f"Potential fighting detected between persons",
                                    metadata={
                                        'person1_id': person1['track_id'],
                                        'person2_id': person2['track_id'],
                                        'distance': distance,
                                        'speed1': speed1,
                                        'speed2': speed2,
                                        'erratic1': changes1,
                                        'erratic2': changes2
                                    }
                                )
                                alerts.append(alert)
                                self.alert_cooldowns[pair_key] = current_time
        
        return alerts

class UnusualMovementDetector:
    """Detects unusual movement patterns using rule-based analysis."""
    
    def __init__(self,
                 speed_threshold_multiplier: float = 2.5,
                 direction_change_threshold: int = 5,
                 loitering_time: float = 15.0,
                 loitering_radius: float = 80.0):
        """
        Initialize unusual movement detector.
        
        Args:
            speed_threshold_multiplier: Multiplier for mean speed to detect fast movement
            direction_change_threshold: Number of direction changes to trigger alert
            loitering_time: Time threshold for loitering detection (seconds)
        """
        self.speed_threshold_multiplier = speed_threshold_multiplier
        self.direction_change_threshold = direction_change_threshold
        self.loitering_time = loitering_time
        self.loitering_radius = loitering_radius
        
        # Track movement patterns
        self.movement_history = defaultdict(list)
        self.speed_baselines = {}
        self.alert_cooldowns = defaultdict(float)
        self.loitering_positions = defaultdict(list)
        self.loitering_start_times = {}
        
    def update(self, tracked_objects: List[Dict], tracker) -> List[Alert]:
        """
        Update unusual movement detection.
        
        Args:
            tracked_objects: Current tracked objects
            tracker: Tracker instance for getting statistics
            
        Returns:
            List of unusual movement alerts
        """
        alerts = []
        current_time = time.time()
        
        # Only analyze people for unusual movement
        people = [obj for obj in tracked_objects if obj['is_person']]
        
        for person in people:
            person_id = person['track_id']
            
            # Skip if in cooldown period
            if current_time - self.alert_cooldowns[person_id] < 10.0:
                continue
            
            # Get movement statistics
            stats = tracker.get_track_statistics(person_id)
            
            if stats.get('track_length', 0) < 30:  # Need enough history
                continue
            
            # Update speed baseline
            avg_speed = stats.get('avg_speed', 0)
            if person_id not in self.speed_baselines:
                self.speed_baselines[person_id] = []
            
            self.speed_baselines[person_id].append(avg_speed)
            if len(self.speed_baselines[person_id]) > 100:
                self.speed_baselines[person_id].pop(0)
            
            # Calculate baseline statistics
            baseline_speeds = self.speed_baselines[person_id]
            if len(baseline_speeds) < 10:
                continue
            
            mean_speed = np.mean(baseline_speeds)
            std_speed = np.std(baseline_speeds)
            
            # Detect unusual speed
            current_speed = stats.get('max_speed', 0)
            speed_threshold = mean_speed + self.speed_threshold_multiplier * std_speed
            
            if current_speed > speed_threshold and current_speed > 50:  # Minimum speed threshold
                alert = Alert(
                    alert_type="unusual_movement_speed",
                    track_id=person_id,
                    confidence=min(0.9, current_speed / speed_threshold * 0.5),
                    timestamp=current_time,
                    bbox=person['bbox'],
                    description=f"Unusually fast movement detected",
                    metadata={
                        'current_speed': current_speed,
                        'baseline_speed': mean_speed,
                        'speed_ratio': current_speed / mean_speed if mean_speed > 0 else 0,
                        'location': person['center']
                    }
                )
                alerts.append(alert)
                self.alert_cooldowns[person_id] = current_time
            
            # Detect erratic movement (direction changes)
            direction_changes = stats.get('direction_changes', 0)
            if direction_changes > self.direction_change_threshold:
                alert = Alert(
                    alert_type="unusual_movement_erratic",
                    track_id=person_id,
                    confidence=min(0.8, direction_changes / (self.direction_change_threshold * 2)),
                    timestamp=current_time,
                    bbox=person['bbox'],
                    description=f"Erratic movement pattern detected",
                    metadata={
                        'direction_changes': direction_changes,
                        'threshold': self.direction_change_threshold,
                        'location': person['center']
                    }
                )
                alerts.append(alert)
                self.alert_cooldowns[person_id] = current_time
            
            # Enhanced loitering detection
            current_pos = person['center']
            
            # Track position history for loitering
            if person_id not in self.loitering_positions:
                self.loitering_positions[person_id] = []
                self.loitering_start_times[person_id] = current_time
            
            self.loitering_positions[person_id].append((current_pos, current_time))
            
            # Keep only recent positions (last 30 seconds for more responsive detection)
            self.loitering_positions[person_id] = [
                (pos, t) for pos, t in self.loitering_positions[person_id]
                if current_time - t <= 30.0
            ]
            
            # Check if person has been in same area for extended time
            if len(self.loitering_positions[person_id]) > 5:  # Reduced threshold
                positions = [pos for pos, _ in self.loitering_positions[person_id]]
                
                # Calculate area coverage (standard deviation of positions)
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                x_std = np.std(x_coords)
                y_std = np.std(y_coords)
                movement_radius = np.sqrt(x_std**2 + y_std**2)
                
                # Check if person stayed in small area for long time
                time_in_area = current_time - self.loitering_start_times[person_id]
                
                # More lenient conditions for loitering detection
                if (movement_radius < self.loitering_radius and 
                    time_in_area > self.loitering_time and
                    avg_speed < 20.0):  # Increased speed threshold
                    
                    if current_time - self.alert_cooldowns[f"loiter_{person_id}"] > 20.0:  # Reduced cooldown
                        alert = Alert(
                            alert_type="loitering",
                            track_id=person_id,
                            confidence=0.8,
                            timestamp=current_time,
                            bbox=person['bbox'],
                            description=f"Loitering detected for {time_in_area:.1f}s",
                            metadata={
                                'duration': time_in_area,
                                'movement_radius': movement_radius,
                                'avg_speed': avg_speed,
                                'location': person['center'],
                                'position_count': len(positions)
                            }
                        )
                        alerts.append(alert)
                        self.alert_cooldowns[f"loiter_{person_id}"] = current_time
                        print(f"LOITERING DETECTED: Person {person_id} stayed in {movement_radius:.1f}px radius for {time_in_area:.1f}s")
            
            # Reset loitering timer if person moves significantly
            if len(self.loitering_positions[person_id]) > 1:
                # Check distance from initial position
                initial_pos = self.loitering_positions[person_id][0][0]
                distance_from_start = np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2)
                if distance_from_start > self.loitering_radius * 1.5:  # Reset if moved far from start
                    self.loitering_start_times[person_id] = current_time
                    self.loitering_positions[person_id] = [(current_pos, current_time)]
        
        return alerts

class AnalyticsEngine:
    """Main analytics engine combining all detection modules."""
    
    def __init__(self):
        """Initialize analytics engine."""
        self.abandonment_detector = AbandonmentDetector()
        self.movement_detector = UnusualMovementDetector()
        self.fighting_detector = FightingDetector()
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        
        print("[INFO] Analytics engine initialized")
    
    def process_frame(self, tracked_objects: List[Dict], tracker) -> List[Alert]:
        """
        Process frame and generate alerts.
        
        Args:
            tracked_objects: Current tracked objects
            tracker: Tracker instance
            
        Returns:
            List of new alerts
        """
        all_alerts = []
        
        # Run abandonment detection
        abandonment_alerts = self.abandonment_detector.update(tracked_objects, tracker)
        all_alerts.extend(abandonment_alerts)
        
        # Run unusual movement detection
        movement_alerts = self.movement_detector.update(tracked_objects, tracker)
        all_alerts.extend(movement_alerts)
        
        # Run fighting detection
        people = [obj for obj in tracked_objects if obj['is_person']]
        if len(people) >= 2:
            fighting_alerts = self.fighting_detector.detect_fighting(people, tracker, time.time())
            all_alerts.extend(fighting_alerts)
        
        # Store alerts in history
        for alert in all_alerts:
            self.alert_history.append(alert)
            alert_key = f"{alert.alert_type}_{alert.track_id}"
            self.active_alerts[alert_key] = alert
        
        return all_alerts
    
    def get_recent_alerts(self, time_window: float = 300.0) -> List[Alert]:
        """Get alerts from recent time window."""
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert.timestamp <= time_window
        ]
        return recent_alerts
    
    def draw_alerts(self, frame: np.ndarray, alerts: List[Alert]) -> np.ndarray:
        """Draw alert information on frame."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Alert colors
        alert_colors = {
            'object_abandonment': (0, 0, 255),      # Red
            'unusual_movement_speed': (0, 165, 255), # Orange
            'unusual_movement_erratic': (255, 0, 255), # Magenta
            'loitering': (0, 255, 255),              # Yellow
            'fighting_detected': (0, 0, 128)         # Dark Red
        }
        
        # Draw alert boxes
        for alert in alerts:
            x1, y1, x2, y2 = alert.bbox
            color = alert_colors.get(alert.alert_type, (255, 255, 255))
            
            # Draw thick alert box
            cv2.rectangle(vis_frame, (x1-5, y1-5), (x2+5, y2+5), color, 4)
            
            # Draw alert label
            label = f"ALERT: {alert.description}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0], y1 - 5), color, -1)
            
            # Label text
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw alert summary
        if alerts:
            alert_summary = f"ACTIVE ALERTS: {len(alerts)}"
            cv2.rectangle(vis_frame, (10, 10), (300, 50), (0, 0, 0), -1)
            cv2.putText(vis_frame, alert_summary, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return vis_frame
