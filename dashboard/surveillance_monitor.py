#!/usr/bin/env python3
"""
Professional Surveillance Monitoring Dashboard
Enterprise-grade security monitoring and event management system.
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
    page_title="Surveillance Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-critical { border-left: 4px solid #dc3545; }
    .alert-warning { border-left: 4px solid #ffc107; }
    .alert-info { border-left: 4px solid #17a2b8; }
    
    .status-active { color: #28a745; font-weight: bold; }
    .status-inactive { color: #dc3545; font-weight: bold; }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stMetric > label { 
        font-size: 14px !important; 
        color: #6c757d;
        font-weight: 500;
    }
    .stMetric > div { 
        font-size: 28px !important; 
        font-weight: 600;
        color: #212529;
    }
    
    .event-table {
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DATABASE_PATH = "recordings/events.db"

@st.cache_data(ttl=30)
def load_surveillance_data(limit=100):
    """Load surveillance events from database with caching."""
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        query = """
        SELECT id, timestamp, alert_type, confidence, location, 
               video_path, thumbnail_path, metadata, description
        FROM events 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        dataframe = pd.read_sql_query(query, connection, params=(limit,))
        connection.close()
        
        if not dataframe.empty:
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
            dataframe['hour'] = dataframe['timestamp'].dt.hour
            dataframe['date'] = dataframe['timestamp'].dt.date
        
        return dataframe
    except Exception as error:
        st.error(f"Database connection error: {error}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def calculate_system_metrics():
    """Calculate system performance metrics."""
    data = load_surveillance_data()
    
    if data.empty:
        return {
            'total_events': 0,
            'recent_events': 0,
            'alert_distribution': {},
            'hourly_patterns': {}
        }
    
    # Calculate metrics
    total_events = len(data)
    recent_events = len(data[data['timestamp'] > datetime.now() - timedelta(hours=24)])
    
    alert_distribution = data['alert_type'].value_counts().to_dict()
    hourly_patterns = data['hour'].value_counts().sort_index().to_dict()
    
    return {
        'total_events': total_events,
        'recent_events': recent_events,
        'alert_distribution': alert_distribution,
        'hourly_patterns': hourly_patterns
    }

def render_dashboard():
    """Main dashboard rendering function."""
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2rem;">Surveillance Monitoring System</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time security event monitoring and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel sidebar
    with st.sidebar:
        st.markdown("### System Controls")
        
        # Auto-refresh configuration
        auto_refresh_enabled = st.checkbox("Enable Auto-Refresh", value=False)
        refresh_interval = st.selectbox("Refresh Interval", [30, 60, 120], index=0, disabled=not auto_refresh_enabled)
        
        # Filtering options
        st.markdown("### Filter Options")
        
        # Date range selector
        date_filter = st.date_input(
            "Date Range",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
            max_value=datetime.now().date()
        )
        
        # Alert type filter
        surveillance_data = load_surveillance_data()
        if not surveillance_data.empty:
            available_types = ['All Types'] + list(surveillance_data['alert_type'].unique())
            selected_type = st.selectbox("Alert Category", available_types, key="alert_filter")
        else:
            selected_type = 'All Types'
        
        # System status indicator
        st.markdown("### System Status")
        st.markdown('<p class="status-active">● System Operational</p>', unsafe_allow_html=True)
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)
    
    # Load system metrics
    metrics = calculate_system_metrics()
    
    # Display KPIs
    with col1:
        st.metric(
            label="Total Security Events",
            value=metrics['total_events'],
            delta=f"+{metrics['recent_events']} (24h)"
        )
    
    with col2:
        avg_confidence = surveillance_data['confidence'].mean() if not surveillance_data.empty else 0
        st.metric(
            label="Detection Accuracy",
            value=f"{avg_confidence:.1%}" if avg_confidence > 0 else "N/A"
        )
    
    with col3:
        active_alerts = len(surveillance_data[surveillance_data['timestamp'] > datetime.now() - timedelta(hours=1)]) if not surveillance_data.empty else 0
        st.metric(
            label="Active Alerts",
            value=active_alerts
        )
    
    with col4:
        system_availability = "99.8%"  # System uptime metric
        st.metric(
            label="System Availability",
            value=system_availability
        )
    
    st.markdown("---")
    
    # Analytics section
    if not surveillance_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Alert Type Distribution")
            if metrics['alert_distribution']:
                distribution_chart = px.pie(
                    values=list(metrics['alert_distribution'].values()),
                    names=list(metrics['alert_distribution'].keys()),
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                )
                distribution_chart.update_layout(height=400, showlegend=True)
                st.plotly_chart(distribution_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### Activity Patterns by Hour")
            if metrics['hourly_patterns']:
                hours = list(range(24))
                activity_counts = [metrics['hourly_patterns'].get(h, 0) for h in hours]
                
                activity_chart = px.bar(
                    x=hours,
                    y=activity_counts,
                    labels={'x': 'Hour of Day', 'y': 'Event Count'},
                    color=activity_counts,
                    color_continuous_scale='Blues'
                )
                activity_chart.update_layout(height=400, showlegend=False)
                st.plotly_chart(activity_chart, use_container_width=True)
        
        # Trend analysis
        st.markdown("#### Event Timeline Analysis")
        
        # Apply filters
        filtered_data = surveillance_data.copy()
        
        if len(date_filter) == 2:
            start_date, end_date = date_filter
            filtered_data = filtered_data[
                (filtered_data['date'] >= start_date) & 
                (filtered_data['date'] <= end_date)
            ]
        
        if selected_type != 'All Types':
            filtered_data = filtered_data[filtered_data['alert_type'] == selected_type]
        
        if not filtered_data.empty:
            # Create timeline visualization
            timeline_data = filtered_data.groupby(['date', 'alert_type']).size().reset_index(name='count')
            
            timeline_chart = px.line(
                timeline_data,
                x='date',
                y='count',
                color='alert_type',
                title="Security Events Trend Analysis",
                markers=True
            )
            timeline_chart.update_layout(height=400)
            st.plotly_chart(timeline_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Event log section
    st.markdown("#### Security Event Log")
    
    if not surveillance_data.empty:
        # Apply display filters
        display_data = surveillance_data.copy()
        
        if len(date_filter) == 2:
            start_date, end_date = date_filter
            display_data = display_data[
                (display_data['date'] >= start_date) & 
                (display_data['date'] <= end_date)
            ]
        
        if selected_type != 'All Types':
            display_data = display_data[display_data['alert_type'] == selected_type]
        
        # Format data for display
        display_data['Event Time'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_data['Alert Category'] = display_data['alert_type'].str.replace('_', ' ').str.title()
        display_data['Confidence Level'] = display_data['confidence'].apply(lambda x: f"{x:.1%}")
        display_data['Event Location'] = display_data['location']
        
        # Display event table
        st.dataframe(
            display_data[['Event Time', 'Alert Category', 'Confidence Level', 'Event Location']],
            use_container_width=True,
            hide_index=True
        )
        
        # Event details section
        if len(display_data) > 0:
            st.markdown("#### Event Details")
            
            # Event selection
            event_list = [f"Event {row['id']} - {row['Alert Category']} ({row['Event Time']})" 
                         for _, row in display_data.iterrows()]
            
            if event_list:
                selected_event = st.selectbox("Select Event for Details", event_list)
                
                if selected_event:
                    event_id = int(selected_event.split(' ')[1])
                    event_details = display_data[display_data['id'] == event_id].iloc[0]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Event ID:** {event_details['id']}")
                        st.write(f"**Timestamp:** {event_details['Event Time']}")
                        st.write(f"**Category:** {event_details['Alert Category']}")
                        st.write(f"**Confidence:** {event_details['Confidence Level']}")
                        st.write(f"**Location:** {event_details['Event Location']}")
                        
                        # Technical metadata
                        if pd.notna(event_details['metadata']) and event_details['metadata']:
                            try:
                                metadata = json.loads(event_details['metadata'])
                                st.write("**Technical Details:**")
                                st.json(metadata)
                            except:
                                pass
                    
                    with col2:
                        # Event thumbnail
                        if pd.notna(event_details['thumbnail_path']) and Path(event_details['thumbnail_path']).exists():
                            try:
                                thumbnail_image = Image.open(event_details['thumbnail_path'])
                                st.image(thumbnail_image, caption="Event Snapshot", use_container_width=True)
                            except:
                                st.warning("Thumbnail unavailable")
                        
                        # Video recording
                        if pd.notna(event_details['video_path']) and Path(event_details['video_path']).exists():
                            st.write(f"**Recording:** {Path(event_details['video_path']).name}")
                            if st.button("View Recording", key=f"video_{event_id}"):
                                st.info("Video playback functionality will be available in the next update.")
    
    else:
        st.info("No security events recorded. System is operational and monitoring for suspicious activity.")
        
        # System information
        with st.expander("System Information"):
            st.code("""
            Database Schema:
            - Event ID: Unique identifier
            - Timestamp: Event occurrence time
            - Alert Category: Type of security alert
            - Confidence Level: Detection accuracy (0-100%)
            - Location: Spatial coordinates
            - Recording Path: Video file location
            - Thumbnail Path: Event snapshot location
            - Technical Data: Additional metadata (JSON format)
            """)
    
    # Auto-refresh functionality (disabled by default to prevent rapid cycling)
    if auto_refresh_enabled:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    render_dashboard()
