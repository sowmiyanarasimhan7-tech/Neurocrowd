"""
NeuroCrowd System Configuration & Safety Benchmarks
"""

# Fruin's Level of Service (LoS) Crowd Density Thresholds (People per Square Meter)
FRUIN_THRESHOLDS = {
    "SAFE": 1.5,          # < 1.5 p/m² : Normal movement
    "MODERATE": 2.5,      # 1.5 - 2.5 p/m² : Restricted walking speed
    "HIGH_RISK": 3.5,     # 2.5 - 3.5 p/m² : Touching contact, surge potential
    "CRITICAL": 3.5       # > 3.5 p/m² : Stampede / Crush hazard
}

# Theme Colors for UI and Risk Overlays
RISK_COLORS = {
    "SAFE": "#00E676",        # Vibrant Green
    "MODERATE": "#FFD600",    # Warning Yellow
    "HIGH_RISK": "#FF9100",   # High Orange
    "CRITICAL": "#FF1744"     # Emergency Red
}

# Default Venue & Calibration Settings
DEFAULT_VENUE_SETTINGS = {
    "venue_name": "Main Central Plaza",
    "total_area_sqm": 600.0,       # Total physical space in square meters
    "obstacle_percentage": 20.0,   # Static obstacles (stage, barricades, booths)
    "grid_rows": 2,
    "grid_cols": 3,
    "confidence_threshold": 0.25,   # YOLO detection confidence threshold
    "head_scaling_factor": 1.15,   # Density estimation factor for overlapping heads
}

# Tactical Authority Action Rules
AUTHORITY_ADVISORIES = {
    "SAFE": [
        "✅ Crowd flow normal. All perimeter gates operating standard intake.",
        "🟢 Maintain regular CCTV monitoring schedule."
    ],
    "MODERATE": [
        "🟡 Alert ground officers in Zone 2 to monitor flow speed.",
        "⚠️ Prepare relief gates for open-access deployment if density reaches 2.2 p/m².",
        "📢 Issue gentle public address announcement to encourage continuous movement."
    ],
    "HIGH_RISK": [
        "🟠 ACTION REQUIRED: Halt incoming pedestrian flow at North & East Entrances.",
        "🚨 Dispatch Rapid Reaction Force (Unit B) to bottleneck sector.",
        "🚪 Open emergency bypass corridors 1 and 4 immediately to relieve pressure."
    ],
    "CRITICAL": [
        "🔴 EMERGENCY STAMPEDE HAZARD: Initiate immediate crowd dispersal protocol!",
        "🚨 Activate all emergency exit gates simultaneously.",
        "🔊 Broadcast high-priority evacuation & directional advisory via PA system.",
        "🚑 Alert onsite medical emergency services to assemble at South Precinct."
    ]
}