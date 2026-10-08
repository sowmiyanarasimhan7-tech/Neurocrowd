"""
NeuroCrowd Computer Vision & Dense Crowd Processing Engine
"""

import cv2
import numpy as np
from config import RISK_COLORS

class CrowdCVEngine:
    def __init__(self, confidence_thresh=0.15):
        self.conf_thresh = confidence_thresh
        self.prev_gray = None
        self.yolo_model = None
        self._init_detector()

    def _init_detector(self):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            self.yolo_model = None
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_people(self, frame, conf_thresh=0.15, dense_head_mode=True):
        h, w = frame.shape[:2]
        boxes = []
        centers = []

        if self.yolo_model is not None:
            results = self.yolo_model(frame, verbose=False, conf=conf_thresh, classes=[0])
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((x1, y1, x2, y2))
                    centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        if dense_head_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            existing_centers = np.array(centers) if len(centers) > 0 else np.empty((0, 2))
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 25 < area < 400:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        if len(existing_centers) > 0:
                            dists = np.linalg.norm(existing_centers - np.array([cx, cy]), axis=1)
                            if np.min(dists) < 18:
                                continue
                        
                        centers.append((cx, cy))
                        boxes.append((cx - 8, cy - 8, cx + 8, cy + 8))

        return boxes, centers

    def compute_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return 1.2, 0.5

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = float(np.mean(mag)) * 5.0
        turbulence = float(np.std(ang)) * 1.5

        return round(avg_mag, 2), round(turbulence, 2)

    def draw_density_heatmap(self, frame, centers, opacity=0.25):
        h, w = frame.shape[:2]
        heatmap_acc = np.zeros((h, w), dtype=np.float32)

        for cx, cy in centers:
            cv2.circle(heatmap_acc, (cx, cy), radius=35, color=1.0, thickness=-1)

        if len(centers) > 0:
            heatmap_acc = cv2.GaussianBlur(heatmap_acc, (77, 77), 0)
            cv2.normalize(heatmap_acc, heatmap_acc, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_acc.astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            alpha = max(0.0, min(1.0, opacity))
            blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap_color, alpha, 0)
            return blended
        return frame

    def draw_zone_grid_overlay(self, frame, zones):
        overlay = frame.copy()
        
        for z in zones:
            x1, y1, x2, y2 = z["bbox"]
            risk = z["risk"]
            density = z["density"]
            count = z["people_count"]
            name = z["name"]
            
            color_hex = RISK_COLORS.get(risk, "#00E676")
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            bgr_color = (b, g, r)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, 2)
            
            banner_h = 28
            cv2.rectangle(overlay, (x1, y1), (x2, y1 + banner_h), (15, 20, 28), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y1 + banner_h), bgr_color, 1)

            label_txt = f"{name} | {count} p ({density} p/m²)"
            cv2.putText(overlay, label_txt, (x1 + 6, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(overlay, (x2 - 12, y1 + 14), 5, bgr_color, -1)

        result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        return result