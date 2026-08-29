"""
NeuroCrowd Spatial Calibration & Net Usable Area Engine
Includes Automatic Perspective Area Estimation & Flexible Zone Naming (Numeric, Alphanumeric, Landmarks)
"""

from config import FRUIN_THRESHOLDS
import numpy as np

def estimate_area_from_video_perspective(frame_width: int, frame_height: int, person_boxes=None) -> float:
    """
    Auto-estimates real-world physical ground area (m²) from camera resolution
    and human perspective height scale.
    """
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
    """
    Computes Net Usable Area by deducting space occupied by static objects.
    """
    obstacle_percentage = max(0.0, min(90.0, obstacle_percentage))
    usable_ratio = (100.0 - obstacle_percentage) / 100.0
    return max(1.0, total_area_sqm * usable_ratio)

def calculate_real_density(people_count: int, net_usable_area_sqm: float) -> float:
    """
    Calculates physical density in people per square meter (p/m²).
    """
    if net_usable_area_sqm <= 0:
        return 0.0
    return round(people_count / net_usable_area_sqm, 2)

def get_fruin_risk_level(density_sqm: float) -> str:
    """
    Evaluates risk level based on Fruin's Level of Service (LoS) benchmarks.
    """
    if density_sqm < FRUIN_THRESHOLDS["SAFE"]:
        return "SAFE"
    elif density_sqm < FRUIN_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif density_sqm < FRUIN_THRESHOLDS["HIGH_RISK"]:
        return "HIGH_RISK"
    else:
        return "CRITICAL"

def generate_zone_grid(frame_width: int, frame_height: int, rows: int = 2, cols: int = 3,
                       total_area_sqm: float = 600.0, obstacle_percentage: float = 20.0,
                       naming_style: str = "numeric", custom_prefix: str = "Zone"):
    """
    Splits camera space into clean logical grid zones with dynamic naming.
    `naming_style`: 'numeric' (Zone 1, Zone 2), 'grid' (Zone A1, A2, B1, B2), or 'landmarks' (Entry, Stage, Exit).
    """
    zones = []
    cell_w = frame_width / cols
    cell_h = frame_height / rows
    num_zones = rows * cols
    
    zone_total_area = total_area_sqm / num_zones
    zone_usable_area = calculate_net_usable_area(zone_total_area, obstacle_percentage)
    
    landmark_names = [
        "North-West Entry", "Main Stage Front", "North-East Plaza",
        "South Corridor", "Central Concourse", "South-East Exit",
        "VIP Sector A", "West Gate Relief", "East Overflow"
    ]
    
    idx = 0
    row_labels = ["A", "B", "C", "D", "E"]
    
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            
            if naming_style == "numeric":
                name = f"{custom_prefix} {idx+1}"
            elif naming_style == "grid":
                r_label = row_labels[r] if r < len(row_labels) else f"R{r+1}"
                name = f"Zone {r_label}{c+1}"
            elif naming_style == "landmarks":
                name = landmark_names[idx] if idx < len(landmark_names) else f"Zone {idx+1}"
            else:
                name = f"Zone {idx+1}"
            
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
    """
    Maps detected person centroids to designated zones and calculates accurate zone density.
    """
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