import cv2
import time
import argparse
from src.detection.person_detector import PersonDetector
from src.analysis.zone_manager import ZoneManager
from src.analysis.motion_analyzer import MotionAnalyzer
from src.prediction.risk_predictor import RiskPredictor
from src.recommendation.engine import RecommendationEngine

def run_pipeline(source=0, show_flow=False):
    """
    Main crowd safety pipeline.
    source: 0 = webcam, or path to video file
    """
    print("[SYSTEM] Initializing modules...")
    detector = PersonDetector(model_path="yolov8n.pt", confidence=0.4)
    zone_mgr = ZoneManager()
    motion_analyzer = MotionAnalyzer()
    risk_predictor = RiskPredictor()
    rec_engine = RecommendationEngine()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    print("[SYSTEM] Pipeline running. Press 'q' to quit.")
    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or cannot read frame.")
            break

        frame_count += 1

        # --- DETECTION ---
        detections = detector.detect(frame)
        total_people = len(detections)

        # --- ZONE ASSIGNMENT ---
        zone_people = zone_mgr.count_people_per_zone(detections, frame)
        zone_summary = zone_mgr.get_zone_summary()

        # --- MOTION ANALYSIS ---
        motion_metrics = motion_analyzer.analyze(frame)

        # --- RISK PREDICTION ---
        overall_risk = risk_predictor.predict(zone_summary, motion_metrics)
        probabilities = risk_predictor.predict_proba(zone_summary, motion_metrics)

        # --- RECOMMENDATIONS (every 30 frames) ---
        recommendations = []
        if frame_count % 30 == 0:
            recommendations = rec_engine.generate(zone_summary, motion_metrics, overall_risk=overall_risk)
            for r in recommendations[-3:]:
                print(r)

        # --- DRAW ---
        display = frame.copy()
        if show_flow:
            display = motion_metrics["flow_frame"]

        display = zone_mgr.draw_zones(display)
        display = detector.draw_detections(display, detections)

        # HUD
        risk_color = {"SAFE": (0,200,0), "WARNING": (0,165,255), "CRITICAL": (0,0,255)}
        color = risk_color.get(overall_risk, (255,255,255))

        cv2.rectangle(display, (0, 0), (400, 110), (0, 0, 0), -1)
        cv2.putText(display, f"TOTAL PEOPLE: {total_people}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display, f"RISK: {overall_risk}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"MOTION: {motion_metrics['avg_magnitude']:.2f} | TURB: {motion_metrics['turbulence']:.2f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(display, f"PANIC: {'YES' if motion_metrics['panic_detected'] else 'NO'}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,0,255) if motion_metrics['panic_detected'] else (0,200,0), 1)

        # FPS
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

        cv2.imshow("Crowd Safety System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[SYSTEM] Shutting down.")
    print("\n--- FINAL RECOMMENDATIONS LOG ---")
    for r in rec_engine.get_log():
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd Safety System")
    parser.add_argument("--source", default="0", help="Video source: 0=webcam or path/to/video.mp4")
    parser.add_argument("--flow", action="store_true", help="Show optical flow visualization")
    args = parser.parse_args()

    source = args.source
    if source != "0" and not source.isnumeric():
        pass  # It's a file path
    else:
        source = int(source)

    run_pipeline(source=source, show_flow=args.flow)