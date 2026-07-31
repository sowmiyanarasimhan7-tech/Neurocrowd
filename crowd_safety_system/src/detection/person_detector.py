import cv2
import torch
from ultralytics import YOLO


class PersonDetector:
    """
    Person Detection Module
    -----------------------
    • Auto GPU detection
    • Works on CPU and NVIDIA GPU
    • Compatible with current project
    • Optimized for hackathon demo
    """

    PERSON_CLASS = 0

    def __init__(
        self,
        model_path="yolov8s.pt",
        confidence=0.45,
        min_box_area=900
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 55)
        print("[Detector]")
        print(f"Device      : {self.device.upper()}")
        print(f"Model       : {model_path}")
        print(f"Confidence  : {confidence}")
        print("=" * 55)

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.confidence = confidence
        self.min_box_area = min_box_area

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS],
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1

                area = width * height

                if area < self.min_box_area:
                    continue

                confidence = float(box.conf.item())

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                        "confidence": confidence,
                        "area": area,
                        "width": width,
                        "height": height,
                    }
                )

        return detections

    def draw_detections(self, frame, detections):

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            conf = det["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.circle(
                frame,
                det["center"],
                4,
                (0, 0, 255),
                -1,
            )

            label = f"Person {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        return frame