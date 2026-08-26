"""
NeuroCrowd Spatial Calibration & Net Usable Area Engine
Includes Automatic Perspective & Real-World Area Estimation from Camera Feed
"""

from config import FRUIN_THRESHOLDS
import numpy as np

def estimate_area_from_video_perspective(frame_width: int, frame_height: int, person_boxes=None) -> float:
    if person_boxes is not None and len(person_boxes) > 0:
        box_heights = [abs(y2 - y1) for (x1, y1, x2, y2) in person_boxes if abs(y2 - y1) > 10]
        if len(box_heights) > 0:
            median_h_px = float(np.median(box_heights))
            meters_per_pixel = 1.7 / max(15.0, median_h_px)
            
            real_width_m = frame_width * meters_per_pixel
            real_height_m = frame_height * meters_per_pixel
            estimated_area = real_width_m * real_height_m
            return round(min(5000.0, max(50.0, estimated_area)), 1)
            
    aspect = frame_width / max(1.0, float(frame_height))
    base_m = 25.0
    return round(base_m * (base_m / aspect), 1)

def calculate_net_usable_area(total_area_sqm: float, obstacle_percentage: float) -> float:
    obstacle_percentage = max(0.0, min(90.0, obstacle_percentage))
    usable_ratio = (100.0 - obstacle_percentage) / 100.0
    return max(1.0, total_area_sqm * usable_ratio)

def calculate_real_density(people_count: int, net_usable_area_sqm: float) -> float:
    if net_usable_area_sqm <= 0:
        return 0.0
    return round(people_count / net_usable_area_sqm, 2)

def get_fruin_risk_level(density_sqm: float) -> str:
    if density_sqm < FRUIN_THRESHOLDS["SAFE"]:
        return "SAFE"
    elif density_sqm < FRUIN_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif density_sqm < FRUIN_THRESHOLDS["HIGH_RISK"]:
        return "HIGH_RISK"
    else:
        return "CRITICAL"

def generate_zone_grid(frame_width: int, frame_height: int, rows: int = 2, cols: int = 3,
                       total_area_sqm: float = 600.0, obstacle_percentage: float = 20.0):
    zones = []
    cell_w = frame_width / cols
    cell_h = frame_height / rows
    num_zones = rows * cols
    
    zone_total_area = total_area_sqm / num_zones
    zone_usable_area = calculate_net_usable_area(zone_total_area, obstacle_percentage)
    
    zone_names = [
        "North-West Entry", "Main Stage Front", "North-East Plaza",
        "South Corridor", "Central Concourse", "South-East Exit",
        "VIP Sector A", "West Gate Relief", "East Overflow"
    ]
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            name = zone_names[idx] if idx < len(zone_names) else f"Zone {idx+1}"
            
            zones.append({
                "id": f"Z{idx+1}",
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "total_area": zone_total_area,
                "usable_area": zone_usable_area,
                "people_count": 0,
                "density": 0.0,
                "risk": "SAFE"
            })
            idx += 1
            
    return zones

def map_detections_to_zones(person_centers, zones, total_area_sqm, obstacle_percentage):
    for z in zones:
        z["people_count"] = 0
        
    for cx, cy in person_centers:
        for z in zones:
            x1, y1, x2, y2 = z["bbox"]
            if x1 <= cx < x2 and y1 <= cy < y2:
                z["people_count"] += 1
                break
                
    for z in zones:
        z["density"] = calculate_real_density(z["people_count"], z["usable_area"])
        z["risk"] = get_fruin_risk_level(z["density"])
        
    return zones