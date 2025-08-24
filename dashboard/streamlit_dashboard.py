#!/usr/bin/env python3
"""
Streamlit Dashboard for AI-Powered Surveillance System
Real-time monitoring and event management interface.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .alert-high { border-left-color: #ff6b6b; }
    .alert-medium { border-left-color: #feca57; }
    .alert-low { border-left-color: #48dbfb; }
    
    /* Improve metric readability and contrast */
    .stMetric > label { font-size: 14px !important; color: #111827 !important; font-weight: 600 !important; }
    .stMetric > div { font-size: 26px !important; color: #0f172a !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 14px !important; font-weight: 600 !important; }
    /* Optional: make success deltas greener and negatives redder */
    [data-testid="stMetricDeltaIcon-Up"] { color: #16a34a !important; }
    [data-testid="stMetricDeltaIcon-Down"] { color: #dc2626 !important; }

    /* Card-like styling for metric blocks */
    [data-testid="stMetric"] {
        background: #f8fafc !important;          /* subtle light background */
        border: 1px solid #e5e7eb !important;     /* light border */
        border-radius: 10px !important;           /* rounded corners */
        padding: 14px 16px !important;            /* internal padding */
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important; /* soft shadow */
    }
    /* Add spacing between metric cards */
    [data-testid="stHorizontalBlock"] > div:has([data-testid="stMetric"]) {
        padding: 4px 6px;  /* outer spacing in columns */
    }
</style>
""", unsafe_allow_html=True)

# Database connection
DB_PATH = "recordings/events.db"

@st.cache_data(ttl=30)
def get_events_data(limit=100):
    """Get events from database with caching."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT id, timestamp, alert_type, confidence, location, 
               video_path, thumbnail_path, metadata
        FROM events 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['date'] = df['timestamp'].dt.date
        
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_system_stats():
    """Get system statistics."""
    df = get_events_data()
    
    if df.empty:
        return {
            'total_events': 0,
            'recent_events': 0,
            'alert_types': {},
            'hourly_distribution': {}
        }
    
    # Calculate stats
    total_events = len(df)
    recent_events = len(df[df['timestamp'] > datetime.now() - timedelta(hours=24)])
    
    alert_types = df['alert_type'].value_counts().to_dict()
    hourly_dist = df['hour'].value_counts().sort_index().to_dict()
    
    return {
        'total_events': total_events,
        'recent_events': recent_events,
        'alert_types': alert_types,
        'hourly_distribution': hourly_dist
    }

def main():
    """Main dashboard function."""
    
    # Header
    st.title("AI-Powered Surveillance Dashboard")
    st.markdown("Real-time monitoring and event management system")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        
        # Filters
        st.subheader("Filters")
        
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
            max_value=datetime.now().date()
        )
        
        # Alert type filter
        df = get_events_data()
        if not df.empty:
            alert_types = ['All'] + list(df['alert_type'].unique())
            selected_alert_type = st.selectbox("Alert Type", alert_types)
        else:
            selected_alert_type = 'All'
        
        # System status
        st.subheader("System Status")
        st.success("System Active")
        st.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get stats
    stats = get_system_stats()
    # Load data for metric deltas
    df_metrics = get_events_data()
    now = datetime.now()
    last_24h = now - timedelta(hours=24)
    prev_24h_start = now - timedelta(hours=48)
    
    # Metrics
    with col1:
        st.metric(
            label="Total Events",
            value=stats['total_events'],
            delta=f"+{stats['recent_events']} (24h)"
        )
    
    with col2:
        # Average confidence (recent vs previous window)
        if not df_metrics.empty:
            recent_conf = df_metrics[df_metrics['timestamp'] > last_24h]['confidence'].mean()
            prev_conf = df_metrics[(df_metrics['timestamp'] <= last_24h) & (df_metrics['timestamp'] > prev_24h_start)]['confidence'].mean()
            avg_confidence = recent_conf if not np.isnan(recent_conf) else 0.0
            if not np.isnan(prev_conf) and not np.isnan(recent_conf):
                delta_conf = (recent_conf - prev_conf) * 100.0
                delta_conf_str = f"{delta_conf:+.1f} pp"
            else:
                delta_conf_str = ""
        else:
            avg_confidence = 0.0
            delta_conf_str = ""
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.1%}" if avg_confidence > 0 else "N/A",
            delta=delta_conf_str
        )
    
    with col3:
        # Active alerts: current hour vs previous hour
        if not df_metrics.empty:
            current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            prev_hour_start = current_hour_start - timedelta(hours=1)
            prev_hour_prev = prev_hour_start - timedelta(hours=1)
            active_alerts = len(df_metrics[df_metrics['timestamp'] >= current_hour_start])
            prev_hour_alerts = len(df_metrics[(df_metrics['timestamp'] >= prev_hour_start) & (df_metrics['timestamp'] < current_hour_start)])
            delta_alerts = active_alerts - prev_hour_alerts
            delta_alerts_str = f"{delta_alerts:+d} vs prev hr"
        else:
            active_alerts = 0
            delta_alerts_str = ""
        st.metric(
            label="Active Alerts",
            value=active_alerts,
            delta=delta_alerts_str
        )
    
    with col4:
        # Uptime: compare to target SLA 99.9%
        system_uptime_val = 99.8  # Mock; replace with real uptime calc if available
        target_sla = 99.9
        delta_uptime = system_uptime_val - target_sla
        st.metric(
            label="System Uptime",
            value=f"{system_uptime_val:.1f}%",
            delta=f"{delta_uptime:+.1f} pp vs 99.9%"
        )
    
    st.divider()
    
    # Charts section
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Events by Type")
            if stats['alert_types']:
                fig_pie = px.pie(
                    values=list(stats['alert_types'].values()),
                    names=list(stats['alert_types'].keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("⏰ Hourly Distribution")
            if stats['hourly_distribution']:
                hours = list(range(24))
                counts = [stats['hourly_distribution'].get(h, 0) for h in hours]
                
                fig_bar = px.bar(
                    x=hours,
                    y=counts,
                    labels={'x': 'Hour of Day', 'y': 'Event Count'},
                    color=counts,
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline chart
        st.subheader("Event Timeline")
        
        # Filter data based on selections
        filtered_df = df.copy()
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) & 
                (filtered_df['date'] <= end_date)
            ]
        
        if selected_alert_type != 'All':
            filtered_df = filtered_df[filtered_df['alert_type'] == selected_alert_type]
        
        if not filtered_df.empty:
            # Group by date and alert type
            timeline_data = filtered_df.groupby(['date', 'alert_type']).size().reset_index(name='count')
            
            fig_timeline = px.line(
                timeline_data,
                x='date',
                y='count',
                color='alert_type',
                title="Events Over Time",
                markers=True
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.divider()
    
    # Recent Events Table
    st.subheader("Recent Events")
    
    if not df.empty:
        # Apply filters to display
        display_df = df.copy()
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            display_df = display_df[
                (display_df['date'] >= start_date) & 
                (display_df['date'] <= end_date)
            ]
        
        if selected_alert_type != 'All':
            display_df = display_df[display_df['alert_type'] == selected_alert_type]
        
        # Format for display
        display_df['Timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Alert Type'] = display_df['alert_type']
        display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['Location'] = display_df['location']
        
        # Display table
        st.dataframe(
            display_df[['Timestamp', 'Alert Type', 'Confidence', 'Location']],
            use_container_width=True,
            hide_index=True
        )
        
        # Event details expander
        if len(display_df) > 0:
            st.subheader("Event Details")
            
            # Select event to view
            event_options = [f"Event {row['id']} - {row['Alert Type']} ({row['Timestamp']})" 
                           for _, row in display_df.iterrows()]
            
            if event_options:
                selected_event = st.selectbox("Select Event to View", event_options)
                
                if selected_event:
                    event_id = int(selected_event.split(' ')[1])
                    event_row = display_df[display_df['id'] == event_id].iloc[0]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Event ID:** {event_row['id']}")
                        st.write(f"**Timestamp:** {event_row['Timestamp']}")
                        st.write(f"**Alert Type:** {event_row['Alert Type']}")
                        st.write(f"**Confidence:** {event_row['Confidence']}")
                        st.write(f"**Location:** {event_row['Location']}")
                        
                        # Metadata
                        if pd.notna(event_row['metadata']) and event_row['metadata']:
                            try:
                                metadata = json.loads(event_row['metadata'])
                                st.write("**Metadata:**")
                                st.json(metadata)
                            except:
                                pass
                    
                    with col2:
                        # Thumbnail
                        if pd.notna(event_row['thumbnail_path']) and Path(event_row['thumbnail_path']).exists():
                            try:
                                image = Image.open(event_row['thumbnail_path'])
                                st.image(image, caption="Event Thumbnail", use_container_width=True)
                            except:
                                st.warning("Could not load thumbnail")
                        
                        # Video link
                        if pd.notna(event_row['video_path']) and Path(event_row['video_path']).exists():
                            st.write(f"**Video:** {Path(event_row['video_path']).name}")
                            if st.button("View Video", key=f"video_{event_id}"):
                                st.info("Video playback feature coming soon!")
    
    else:
        st.info("No events found. The surveillance system may be starting up or no alerts have been generated yet.")
        
        # Show sample data structure
        with st.expander("Expected Data Structure"):
            st.code("""
            Events Table Schema:
            - id: Event ID
            - timestamp: Event timestamp
            - alert_type: Type of alert (abandonment, unusual_movement, etc.)
            - confidence: Detection confidence (0.0 - 1.0)
            - location: Event location coordinates
            - video_path: Path to recorded video clip
            - thumbnail_path: Path to event thumbnail
            - metadata: Additional event data (JSON)
            """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
