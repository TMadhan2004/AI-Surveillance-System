#!/usr/bin/env python3
"""
Database Initialization Script
Creates the events table and sets up the database structure.
"""

import sqlite3
import os
from pathlib import Path

def init_database():
    """Initialize the surveillance database."""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Database path
    db_path = data_dir / "surveillance.db"
    
    print(f"🗄️ Initializing database: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
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
            thumbnail_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON events(timestamp)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_alert_type 
        ON events(alert_type)
    ''')
    
    # Commit changes
    conn.commit()
    
    # Verify table creation
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        print("[INFO] Events table created successfully")
        
        # Show table schema
        cursor.execute("PRAGMA table_info(events)")
        columns = cursor.fetchall()
        print("[INFO] Table schema:")
        for col in columns:
            print(f"   • {col[1]} ({col[2]})")
    else:
        print("[ERROR] Failed to create events table")
    
    # Close connection
    conn.close()
    
    print("[INFO] Database initialization complete")

if __name__ == "__main__":
    init_database()
