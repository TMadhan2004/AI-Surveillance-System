#!/usr/bin/env python3
"""
Production-Grade AI Surveillance System
YOLOv5 + DeepSORT + Analytics + Recording + Dashboard
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.detector import YOLODetector
from core.tracker import SurveillanceTracker
from core.analytics import AnalyticsEngine
from core.recorder import EventRecorder, AlertManager

class SurveillanceSystem:
    """Main surveillance system orchestrating all components."""
    
    def __init__(self, 
                 input_source=0,
                 model_size='s',
                 output_dir='recordings'):
        """
        Initialize surveillance system.
        
        Args:
            input_source: Camera index, video file, or RTSP URL
            model_size: YOLOv5 model size ('s', 'm', 'l', 'x')
            output_dir: Directory for recordings and data
        """
        print("[INFO] INITIALIZING PRODUCTION SURVEILLANCE SYSTEM")
        print("=" * 60)
        
        # Initialize components
        self.detector = YOLODetector(model_size=model_size, conf_threshold=0.5)
        self.tracker = SurveillanceTracker(max_age=30, n_init=3)
        self.analytics = AnalyticsEngine()
        self.recorder = EventRecorder(output_dir=output_dir)
        self.alert_manager = AlertManager(self.recorder)
        
        # Video input
        self.input_source = input_source
        self.cap = None
        
        # System state
        self.running = False
        self.fps_counter = []
        self.frame_count = 0
        
        # Performance metrics
        self.metrics = {
            'total_detections': 0,
            'total_tracks': 0,
            'total_alerts': 0,
            'avg_fps': 0
        }
        
        print("[INFO] All components initialized successfully")
    
    def start(self):
        """Start the surveillance system."""
        print(f"\n[INFO] Starting surveillance on: {self.input_source}")
        
        # Open video source
        self.cap = cv2.VideoCapture(self.input_source)
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open video source: {self.input_source}")
            return
        
        # Set camera properties for better performance
        if isinstance(self.input_source, int):  # Webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        print("[INFO] System Status:")
        print("  • Press 'q' to quit")
        print("  • Press 'r' to toggle recording")
        print("  • Press 's' to save current frame")
        print("[INFO] Event recorder initialized - Output: recordings")
        print("  • Press 'i' to show system info")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] Stopping surveillance system...")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop."""
        while self.running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Failed to read frame")
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            # Add frame to recorder buffer
            self.recorder.add_frame(frame, current_time)
            
            # Detection phase
            detections = self.detector.detect(frame)
            self.metrics['total_detections'] += len(detections)
            
            # Tracking phase
            tracked_objects = self.tracker.update(detections, frame)
            self.metrics['total_tracks'] = len(self.tracker.track_history)
            
            # Analytics phase
            alerts = self.analytics.process_frame(tracked_objects, self.tracker)
            if alerts:
                self.metrics['total_alerts'] += len(alerts)
                self.alert_manager.process_alerts(alerts, frame)
            
            # Update recordings
            self.alert_manager.update_recordings(frame)
            
            # Visualization
            vis_frame = self._create_visualization(frame, detections, tracked_objects, alerts)
            
            # Display
            cv2.imshow('Production AI Surveillance System', vis_frame)
            
            # Performance tracking
            loop_time = time.time() - loop_start
            self.fps_counter.append(1.0 / loop_time if loop_time > 0 else 0)
            if len(self.fps_counter) > 30:
                self.fps_counter.pop(0)
            
            self.metrics['avg_fps'] = np.mean(self.fps_counter)
            
            # Print status every 5 seconds
            if self.frame_count % 150 == 0:  # Assuming 30 FPS
                self._print_status(len(detections), len(tracked_objects), len(alerts))
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame(vis_frame)
            elif key == ord('i'):
                self._print_detailed_info()
            elif key == ord('r'):
                self.recorder.toggle_recording()
    
    def _create_visualization(self, frame, detections, tracked_objects, alerts):
        """Create comprehensive visualization."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Draw detections
        vis_frame = self.detector.draw_detections(vis_frame, detections)
        
        # Draw tracks
        vis_frame = self.tracker.draw_tracks(vis_frame, tracked_objects)
        
        # Draw alerts
        vis_frame = self.analytics.draw_alerts(vis_frame, alerts)
        
        # System header
        header_height = 80
        cv2.rectangle(vis_frame, (0, 0), (w, header_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(vis_frame, "PRODUCTION AI SURVEILLANCE SYSTEM", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # System stats
        stats_text = f"FPS: {self.metrics['avg_fps']:.1f} | Detections: {len(detections)} | Tracks: {len(tracked_objects)} | Alerts: {len(alerts)}"
        cv2.putText(vis_frame, stats_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recording indicator
        if self.recorder.is_recording:
            cv2.circle(vis_frame, (w-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(vis_frame, "REC", (w-60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Frame counter
        cv2.putText(vis_frame, f"Frame: {self.frame_count}", (w-150, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def _print_status(self, num_detections, num_tracks, num_alerts):
        """Print system status."""
        print(f"[INFO] Frame {self.frame_count}: "
              f"Det: {num_detections}, Tracks: {num_tracks}, Alerts: {num_alerts}, "
              f"FPS: {self.metrics['avg_fps']:.1f}")
    
    def _print_detailed_info(self):
        """Print detailed system information."""
        print("\n" + "="*60)
        print("[INFO] DETAILED SYSTEM INFORMATION")
        print("="*60)
        print(f"[INFO] Performance Metrics:")
        print(f"  • Average FPS: {self.metrics['avg_fps']:.2f}")
        print(f"[INFO] Target classes: {list(self.detector.target_classes.values())}")
        print(f"  • Total Frames: {self.frame_count}")
        print(f"  • Total Detections: {self.metrics['total_detections']}")
        print(f"  • Active Tracks: {self.metrics['total_tracks']}")
        print(f"  • Total Alerts: {self.metrics['total_alerts']}")
        
        # Alert summary
        summary = self.alert_manager.get_alert_summary()
        print(f"\n[INFO] Alert Summary (Last Hour):")
        print(f"  • Total Alerts: {summary['total_alerts']}")
        for alert_type, count in summary['alert_types'].items():
            print(f"  • {alert_type}: {count}")
        
        print("="*60)
    
    def _save_frame(self, frame):
        """Save current frame."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Frame saved: {filename}")
    
    def stop(self):
        """Stop the surveillance system."""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        cv2.destroyAllWindows()
        
        print(f"\n[INFO] FINAL SYSTEM STATISTICS:")
        print(f"  • Total Runtime: {self.frame_count / max(self.metrics['avg_fps'], 1):.1f} seconds")
        print(f"  • Average FPS: {self.metrics['avg_fps']:.2f}")
        print(f"  • Total Detections: {self.metrics['total_detections']}")
        print(f"  • Total Alerts: {self.metrics['total_alerts']}")
        print("[INFO] Surveillance system stopped")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production AI Surveillance System')
    parser.add_argument('--input', '-i', default=0, 
                       help='Input source (camera index, video file, or RTSP URL)')
    parser.add_argument('--model', '-m', default='s', choices=['s', 'm', 'l', 'x'],
                       help='YOLOv5 model size')
    parser.add_argument('--output', '-o', default='recordings',
                       help='Output directory for recordings')
    
    args = parser.parse_args()
    
    # Convert camera index to int if numeric
    input_source = args.input
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)
    
    # Create and start surveillance system
    system = SurveillanceSystem(
        input_source=input_source,
        model_size=args.model,
        output_dir=args.output
    )
    
    system.start()

if __name__ == "__main__":
    main()
