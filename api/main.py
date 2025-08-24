#!/usr/bin/env python3
"""
FastAPI Backend for AI-Powered Surveillance Dashboard
Provides REST API endpoints for monitoring and managing the surveillance system.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sqlite3
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from pathlib import Path
import cv2
import base64
import threading
import time

app = FastAPI(title="AI Surveillance API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = "recordings/events.db"
RECORDINGS_PATH = "recordings"

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI-Powered Surveillance API", "status": "active"}

@app.get("/api/events")
async def get_events(limit: int = 50, offset: int = 0, alert_type: Optional[str] = None):
    """Get surveillance events with pagination."""
    try:
        conn = get_db_connection()
        
        query = """
        SELECT id, timestamp, alert_type, confidence, location, 
               video_path, thumbnail_path, metadata
        FROM events 
        """
        params = []
        
        if alert_type:
            query += " WHERE alert_type = ?"
            params.append(alert_type)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        events = []
        
        for row in cursor.fetchall():
            event = dict(row)
            if event['metadata']:
                event['metadata'] = json.loads(event['metadata'])
            events.append(event)
        
        conn.close()
        return {"events": events, "total": len(events)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/events/{event_id}")
async def get_event(event_id: int):
    """Get specific event details."""
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Event not found")
        
        event = dict(row)
        if event['metadata']:
            event['metadata'] = json.loads(event['metadata'])
        
        return event
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get surveillance system statistics."""
    try:
        conn = get_db_connection()
        
        # Total events
        cursor = conn.execute("SELECT COUNT(*) as total FROM events")
        total_events = cursor.fetchone()['total']
        
        # Events by type
        cursor = conn.execute("""
            SELECT alert_type, COUNT(*) as count 
            FROM events 
            GROUP BY alert_type
        """)
        events_by_type = {row['alert_type']: row['count'] for row in cursor.fetchall()}
        
        # Recent events (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        cursor = conn.execute(
            "SELECT COUNT(*) as recent FROM events WHERE timestamp > ?",
            (yesterday.isoformat(),)
        )
        recent_events = cursor.fetchone()['recent']
        
        # Events by hour (last 24 hours)
        cursor = conn.execute("""
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM events 
            WHERE timestamp > ?
            GROUP BY hour
            ORDER BY hour
        """, (yesterday.isoformat(),))
        
        hourly_stats = {f"{i:02d}": 0 for i in range(24)}
        for row in cursor.fetchall():
            hourly_stats[row['hour']] = row['count']
        
        conn.close()
        
        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "recent_events": recent_events,
            "hourly_stats": hourly_stats,
            "system_status": "active"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve video files."""
    video_path = Path(RECORDINGS_PATH) / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/api/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """Serve thumbnail images."""
    thumbnail_path = Path(RECORDINGS_PATH) / filename
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.delete("/api/events/{event_id}")
async def delete_event(event_id: int):
    """Delete an event and its associated files."""
    try:
        conn = get_db_connection()
        
        # Get event details first
        cursor = conn.execute(
            "SELECT video_path, thumbnail_path FROM events WHERE id = ?",
            (event_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Delete files
        if row['video_path'] and os.path.exists(row['video_path']):
            os.remove(row['video_path'])
        
        if row['thumbnail_path'] and os.path.exists(row['thumbnail_path']):
            os.remove(row['thumbnail_path'])
        
        # Delete from database
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
        conn.close()
        
        return {"message": "Event deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic system status
            stats = await get_stats()
            await websocket.send_text(json.dumps({
                "type": "stats_update",
                "data": stats
            }))
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/alerts")
async def create_alert(alert_data: dict):
    """Create a new alert (called by surveillance system)."""
    try:
        # Broadcast to connected clients
        await manager.broadcast({
            "type": "new_alert",
            "data": alert_data
        })
        
        return {"message": "Alert broadcasted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for frontend
if os.path.exists("dashboard/build"):
    app.mount("/static", StaticFiles(directory="dashboard/build/static"), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React frontend."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        
        index_path = "dashboard/build/index.html"
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="Frontend not built")

if __name__ == "__main__":
    import uvicorn
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs(RECORDINGS_PATH, exist_ok=True)
    
    print("[INFO] Starting AI Surveillance API Server...")
    print("[INFO] Dashboard: http://localhost:8000")
    print("[INFO] API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
