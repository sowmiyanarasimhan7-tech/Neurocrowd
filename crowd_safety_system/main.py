import argparse
import time
from collections import deque

import cv2

from src.detection.person_detector import PersonDetector
from src.tracking.tracker import SimpleTracker
from src.analysis.zone_manager import ZoneManager
from src.analysis.motion_analyzer import MotionAnalyzer
from src.prediction.risk_predictor import RiskPredictor
from src.recommendation.engine import RecommendationEngine
from src.dashboard.ui import compose_display

# ---- display sizing -------------------------------------------------------
# Video area + sidebar. If your screen is smaller, shrink these -- the
# window itself will resize accordingly, no other code needs to change.
VIDEO_W = 1180
VIDEO_H = 780
SIDEBAR_W = 420
WINDOW_W = VIDEO_W + SIDEBAR_W
WINDOW_H = VIDEO_H

RISK_TO_INDEX = {"SAFE": 0.0, "WARNING": 50.0, "CRITICAL": 100.0}


def run_pipeline(source="data/videos/crowd_test.mp4", show_flow=False):

    detector = PersonDetector(model_path="yolov8s.pt", confidence=0.25)
    tracker = SimpleTracker(max_distance=70, max_misses=8, min_hits=2)
    zone_manager = ZoneManager(config_path="zones/configs/venue_config.yaml")
    motion_analyzer = MotionAnalyzer(history_frames=30, flow_scale=0.5)
    risk_predictor = RiskPredictor(model_path="models/risk_model.pkl")
    rec_engine = RecommendationEngine(config_path="zones/configs/venue_config.yaml")

    risk_history = deque(maxlen=90)  # ~ a few seconds of danger-index history for the sparkline

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print("[SYSTEM] Pipeline running. Press 'q' to quit, 'f' to toggle flow view.")

    cv2.namedWindow("NeuroCrowd - Crowd Safety System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NeuroCrowd - Crowd Safety System", WINDOW_W, WINDOW_H)

    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    while True:

        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or cannot read frame.")
            break

        # ------------------------------------------------------- detect + track
        detections = detector.detect(frame)
        detections, confirmed_count = tracker.update(detections)

        # ------------------------------------------------------- zones
        # Only CONFIRMED tracks (seen 2+ consecutive frames) count toward
        # zone occupancy -- this is the fix for single-frame flicker
        # causing inaccurate counts.
        zone_manager.count_people_per_zone(detections, frame=frame, confirmed_only=True)
        zone_summary = zone_manager.get_zone_summary()
        total_people = zone_manager.get_total_people()

        # ------------------------------------------------------- motion (global + per-zone)
        motion_metrics = motion_analyzer.analyze(frame)
        zone_motion = motion_analyzer.analyze_zone_motion(zone_manager.zones)

        # ------------------------------------------------------- risk (forecast)
        overall_risk = risk_predictor.predict(zone_summary, motion_metrics, zone_motion=zone_motion)
        proba = risk_predictor.predict_proba(zone_summary, motion_metrics, zone_motion=zone_motion)

        danger_index = proba["SAFE"] * 0 + proba["WARNING"] * 50 + proba["CRITICAL"] * 100
        risk_history.append(danger_index)

        # ------------------------------------------------------- recommendations
        recs = rec_engine.generate(
            zone_summary,
            motion_metrics,
            audio_panic=False,
            overall_risk=overall_risk,
        )

        # ------------------------------------------------------- draw
        display_frame = motion_metrics["flow_frame"] if show_flow else frame
        display_frame = zone_manager.draw_zones(display_frame)
        display_frame = detector.draw_detections(display_frame, detections)

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_time) if now > fps_time else fps
            fps_time = now

        final_canvas = compose_display(
            display_frame,
            sidebar_width=SIDEBAR_W,
            video_target_w=VIDEO_W,
            video_target_h=VIDEO_H,
            zone_summary=zone_summary,
            overall_risk=overall_risk,
            proba=proba,
            motion_metrics=motion_metrics,
            zone_motion=zone_motion,
            recs=recs,
            total_people=total_people,
            fps=fps,
            risk_history=risk_history,
        )

        cv2.imshow("NeuroCrowd - Crowd Safety System", final_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[SYSTEM] Quit requested.")
            break
        elif key == ord("f"):
            show_flow = not show_flow

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/videos/crowd_test.mp4",
                         help="Video file path, or 0 for webcam")
    parser.add_argument("--flow", action="store_true", help="Show optical flow overlay instead of raw frame")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_pipeline(source=source, show_flow=args.flow)
