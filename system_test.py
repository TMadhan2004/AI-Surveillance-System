#!/usr/bin/env python3
"""
Complete System Test for AI-Powered Surveillance
Tests the full pipeline: Detection -> Tracking -> Analytics -> Recording -> Dashboard
"""

import subprocess
import time
import threading
import webbrowser
import os
import signal
import sys
from pathlib import Path

class SystemTester:
    """Comprehensive system tester."""
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        
    def start_api_server(self):
        """Start FastAPI backend server."""
        print("[INFO] Starting FastAPI backend server...")
        
        api_process = subprocess.Popen([
            sys.executable, "api/main.py"
        ], cwd=self.base_dir)
        
        self.processes.append(("API Server", api_process))
        
        # Wait for server to start
        time.sleep(3)
        print("[INFO] API Server started on http://localhost:8000")
        
    def start_dashboard(self):
        """Start Streamlit dashboard."""
        print("[INFO] Starting Streamlit dashboard...")
        
        dashboard_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/surveillance_monitor.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ]
        
        dashboard_process = subprocess.Popen(dashboard_cmd, cwd=self.base_dir)
        
        self.processes.append(("Dashboard", dashboard_process))
        
        # Wait for dashboard to start
        time.sleep(5)
        print("[INFO] Dashboard started on http://localhost:8501")
        
    def start_surveillance(self):
        """Start main surveillance system."""
        print("[INFO] Starting surveillance system...")
        
        surveillance_process = subprocess.Popen([
            sys.executable, "surveillance_system.py",
            "--model", "s", "--input", "0"
        ], cwd=self.base_dir)
        
        self.processes.append(("Surveillance", surveillance_process))
        print("[INFO] Surveillance system started")
        
    def open_browsers(self):
        """Open browser tabs for monitoring."""
        print("[INFO] Opening browser interfaces...")
        
        # Wait a bit for services to fully start
        time.sleep(2)
        
        try:
            # Open API docs
            webbrowser.open("http://localhost:8000/docs")
            print("[INFO] Opened API documentation")
            
            # Open dashboard
            webbrowser.open("http://localhost:8501")
            print("[INFO] Opened surveillance dashboard")
            
        except Exception as e:
            print(f"[WARNING] Could not open browsers: {e}")
            print("[INFO] Manual URLs:")
            print("   API Docs: http://localhost:8000/docs")
            print("   Dashboard: http://localhost:8501")
    
    def monitor_system(self):
        """Monitor system status."""
        print("\n" + "="*60)
        print("[INFO] SYSTEM MONITORING ACTIVE")
        print("="*60)
        print("[INFO] Dashboard: http://localhost:8501")
        print("[INFO] API Docs: http://localhost:8000/docs")
        print("[INFO] Surveillance: Running with camera input")
        print("[INFO] Instructions:")
        print("   • Move objects in front of camera to trigger detections")
        print("   • Leave objects stationary to test abandonment detection")
        print("   • Move quickly to test unusual movement detection")
        print("   • Check dashboard for real-time event monitoring")
        print("   • Press Ctrl+C to stop all services")
        print("="*60)
        
        try:
            while True:
                # Check if processes are still running
                running_processes = []
                for name, process in self.processes:
                    if process.poll() is None:
                        running_processes.append(name)
                    else:
                        print(f"[WARNING] {name} process stopped")
                
                if running_processes:
                    print(f"[INFO] Running: {', '.join(running_processes)}")
                else:
                    print("[INFO] All processes stopped")
                    break
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n[INFO] Stopping system...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up all processes."""
        print("[INFO] Cleaning up processes...")
        
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    print(f"[INFO] Stopping {name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"[INFO] Force killing {name}...")
                        process.kill()
                        
            except Exception as e:
                print(f"[WARNING] Error stopping {name}: {e}")
        
        print("[INFO] All processes stopped")
    
    def run_complete_test(self):
        """Run complete system test."""
        print("[INFO] STARTING COMPLETE SURVEILLANCE SYSTEM TEST")
        print("="*60)
        
        try:
            # Start services in order
            self.start_api_server()
            self.start_dashboard()
            
            # Open browser interfaces
            self.open_browsers()
            
            # Start surveillance (this will run in foreground)
            print("\n[INFO] Starting surveillance system...")
            print("[INFO] Note: Surveillance will run in interactive mode")
            print("   Press 'q' in surveillance window to stop")
            
            # Run surveillance in a separate thread so we can monitor
            surveillance_thread = threading.Thread(target=self.start_surveillance)
            surveillance_thread.daemon = True
            surveillance_thread.start()
            
            # Monitor system
            self.monitor_system()
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
        finally:
            self.cleanup()

def main():
    """Main test function."""
    
    # Check if required files exist
    required_files = [
        "surveillance_system.py",
        "api/main.py", 
        "dashboard/surveillance_monitor.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[ERROR] Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return
    
    # Check if camera is available
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Camera not available. System will still start but may not detect objects.")
    else:
        print("[INFO] Camera detected")
    cap.release()
    
    # Run test
    tester = SystemTester()
    tester.run_complete_test()

if __name__ == "__main__":
    main()
