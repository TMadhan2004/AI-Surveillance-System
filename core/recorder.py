#!/usr/bin/env python3
"""
Event Recording and Alerting System
Handles video clip recording, event logging, and notifications.
"""

import cv2
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
import threading
import time
from pathlib import Path

from .analytics import Alert

class EventRecorder:
    """Records video clips and manages event data."""
    
    def __init__(self, 
                 output_dir: str = "recordings",
                 buffer_seconds: int = 10,
                 fps: int = 30,
                 max_clip_duration: int = 20):
        """
        Initialize event recorder.
        
        Args:
            output_dir: Directory to save recordings
            buffer_seconds: Seconds of pre-buffer to keep
            fps: Frames per second for recording
            max_clip_duration: Maximum clip duration in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_clip_duration = max_clip_duration
        
        # Frame buffer for pre-recording
        buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Recording state
        self.is_recording = False
        self.current_recording = None
        self.recording_start_time = None
        self.recording_writer = None
        
        # Database setup
        self.db_path = self.output_dir / "events.db"
        self._init_database()
        
        print(f"[INFO] Event recorder initialized - Output: {self.output_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for events."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                track_id INTEGER,
                confidence REAL,
                location TEXT,
                description TEXT,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                metadata TEXT,
                video_path TEXT,
                thumbnail_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer."""
        self.frame_buffer.append({
            'frame': frame.copy(),
            'timestamp': timestamp
        })
    
    def start_recording(self, alert: Alert) -> str:
        """
        Start recording for an alert.
        
        Args:
            alert: Alert that triggered recording
            
        Returns:
            Path to the recording file
        """
        # Always save alert to database, regardless of recording status
        # Generate filename for potential video
        timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{alert.alert_type}_{alert.track_id}_{timestamp_str}.mp4"
        video_path = self.output_dir / filename
        
        # Get sample frame for thumbnail
        sample_frame = None
        if self.frame_buffer:
            sample_frame = self.frame_buffer[-1]['frame']
        
        # Save thumbnail if we have a frame
        thumbnail_path = None
        if sample_frame is not None:
            thumbnail_path = self._save_thumbnail(sample_frame, alert)
        
        # Always save event to database
        self._save_event_to_db(alert, str(video_path), thumbnail_path)
        
        # Only start video recording if not already recording
        if self.is_recording:
            return str(video_path)  # Return path even if not recording video
        
        # Get frame dimensions from buffer
        if not self.frame_buffer:
            return str(video_path)  # Still return path for database entry
        
        sample_frame = self.frame_buffer[-1]['frame']
        h, w = sample_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
        
        # Write pre-buffer frames (create copy to avoid mutation during iteration)
        buffer_copy = list(self.frame_buffer)
        frame_count = 0
        for frame_data in buffer_copy:
            out.write(frame_data['frame'])
            frame_count += 1
        
        self.is_recording = True
        self.current_recording = str(video_path)
        self.recording_start_time = time.time()
        
        print(f"[INFO] Started recording: {filename}")
        return str(video_path)
    
    def add_recording_frame(self, frame: np.ndarray):
        """Add frame to current recording."""
        if self.is_recording and self.recording_writer:
            self.recording_writer.write(frame)
            
            # Check if recording should stop
            recording_duration = time.time() - self.recording_start_time
            if recording_duration >= self.max_clip_duration:
                self.stop_recording()
    
    def stop_recording(self):
        """Stop current recording."""
        if self.is_recording and self.recording_writer:
            self.recording_writer.release()
            self.recording_writer = None
            self.is_recording = False
            
            print(f"⏹️ Stopped recording: {os.path.basename(self.current_recording)}")
            self.current_recording = None
            self.recording_start_time = None
    
    def _save_thumbnail(self, frame: np.ndarray, alert: Alert) -> str:
        """Save thumbnail for alert."""
        timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y%m%d_%H%M%S")
        thumbnail_filename = f"thumb_{alert.alert_type}_{alert.track_id}_{timestamp_str}.jpg"
        thumbnail_path = self.output_dir / thumbnail_filename
        
        # Crop to alert bbox if available
        if alert.bbox:
            x1, y1, x2, y2 = alert.bbox
            # Expand bbox slightly
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            # Validate bbox dimensions
            if x2 > x1 and y2 > y1:
                thumbnail = frame[y1:y2, x1:x2]
            else:
                thumbnail = frame
        else:
            thumbnail = frame
        
        # Validate thumbnail is not empty
        if thumbnail is None or thumbnail.size == 0:
            print(f"[WARNING] Empty thumbnail for alert {alert.alert_type}, using full frame")
            thumbnail = frame
        
        # Final validation before saving
        if thumbnail is not None and thumbnail.size > 0:
            cv2.imwrite(str(thumbnail_path), thumbnail)
        else:
            print(f"[ERROR] Failed to save thumbnail for alert {alert.alert_type}")
            return None
        return str(thumbnail_path)
    
    def _save_event_to_db(self, alert: Alert, video_path: str, thumbnail_path: str):
        """Save event to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        x1, y1, x2, y2 = alert.bbox if alert.bbox else (0, 0, 0, 0)
        
        # Create location string from bbox
        location = f"({x1},{y1},{x2},{y2})" if alert.bbox else "unknown"
        
        # Convert timestamp to ISO format
        timestamp_str = datetime.fromtimestamp(alert.timestamp).isoformat()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        # Convert metadata to ensure JSON serialization
        serializable_metadata = convert_numpy_types(alert.metadata)
        
        cursor.execute('''
            INSERT INTO events (
                timestamp, alert_type, track_id, confidence, location, description,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, metadata, video_path, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp_str, alert.alert_type, alert.track_id, alert.confidence,
            location, alert.description, x1, y1, x2, y2, json.dumps(serializable_metadata),
            video_path, thumbnail_path
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_events(self, hours: int = 24) -> List[Dict]:
        """Get recent events from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours * 3600)
        
        cursor.execute('''
            SELECT * FROM events 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return events

class AlertManager:
    """Manages alert processing and notifications."""
    
    def __init__(self, recorder: EventRecorder):
        """Initialize alert manager."""
        self.recorder = recorder
        self.active_recordings = {}
        self.alert_queue = deque()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.processing_thread.start()
        
        print("[INFO] Alert manager initialized")
    
    def process_alerts(self, alerts: List[Alert], current_frame: np.ndarray):
        """Process new alerts."""
        for alert in alerts:
            # Add to queue for processing
            self.alert_queue.append({
                'alert': alert,
                'frame': current_frame.copy(),
                'timestamp': time.time()
            })
    
    def _process_alerts(self):
        """Background thread for processing alerts."""
        while True:
            if self.alert_queue:
                alert_data = self.alert_queue.popleft()
                alert = alert_data['alert']
                frame = alert_data['frame']
                
                # Record every alert (not just first per track)
                video_path = self.recorder.start_recording(alert)
                if video_path:
                    # Use timestamp to make each recording unique
                    track_key = f"{alert.alert_type}_{alert.track_id}_{int(alert.timestamp)}"
                    self.active_recordings[track_key] = {
                        'path': video_path,
                        'start_time': time.time()
                    }
                
                # Send notification (placeholder)
                self._send_notification(alert)
            
            time.sleep(0.1)
    
    def _send_notification(self, alert: Alert):
        """Send notification for alert (placeholder)."""
        print(f"ALERT: {alert.description} (Track ID: {alert.track_id})")
    
    def update_recordings(self, current_frame: np.ndarray):
        """Update active recordings."""
        current_time = time.time()
        
        # Clean up old recordings
        to_remove = []
        for track_key, recording_info in self.active_recordings.items():
            if current_time - recording_info['start_time'] > self.recorder.max_clip_duration:
                to_remove.append(track_key)
        
        for key in to_remove:
            del self.active_recordings[key]
        
        # Add frame to current recording
        if self.recorder.is_recording:
            self.recorder.add_recording_frame(current_frame)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts."""
        events = self.recorder.get_recent_events(hours=1)
        
        summary = {
            'total_alerts': len(events),
            'alert_types': {},
            'recent_alerts': events[:10]  # Last 10 alerts
        }
        
        for event in events:
            alert_type = event['alert_type']
            summary['alert_types'][alert_type] = summary['alert_types'].get(alert_type, 0) + 1
        
        return summary
