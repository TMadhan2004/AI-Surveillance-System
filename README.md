AI-Powered Surveillance System

A production-grade real-time surveillance platform using YOLOv5 object detection, DeepSORT tracking, and behavioral analytics.

Features
- Real-time object detection (YOLOv5)
- Multi-object tracking with persistent IDs (DeepSORT)
- Behavioral analytics: abandonment, unusual movement, loitering
- Event recording: MP4 clips + thumbnails + metadata
- Live dashboard (Streamlit) with charts and filters
- REST API (FastAPI) for integration

Project Structure
ai-surveillance/
├── surveillance_system.py           # Main surveillance application
├── system_test.py                   # Complete system test runner
├── requirements.txt                 # Python dependencies
├── core/
│   ├── detector.py                  # YOLOv5 detection engine
│   ├── tracker.py                   # DeepSORT tracking system
│   ├── analytics.py                 # Behavior analysis engine
│   └── recorder.py                  # Event recording system
├── api/
│   └── main.py                      # FastAPI server
├── dashboard/
│   └── streamlit_dashboard.py       # Monitoring dashboard
├── recordings/                      # Clips and thumbnails
├── data/                            # Databases and outputs
└── models/                          # Model weights

Quick Start
Complete system test (API + Dashboard + Surveillance):
python system_test.py


Run individual components:
# Surveillance only
python surveillance_system.py --model s --input 0

# Dashboard only
streamlit run dashboard/streamlit_dashboard.py

# API server only
python api/main.py

## Interfaces
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
