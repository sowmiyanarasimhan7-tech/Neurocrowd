import cv2
import numpy as np
from collections import deque

class MotionAnalyzer:
    def __init__(self, history_frames=30):
        self.prev_gray = None
        self.flow_history = deque(maxlen=history_frames)
        self.panic_threshold = 8.0      # Flow magnitude threshold
        self.turbulence_threshold = 3.5  # Std deviation threshold

    def analyze(self, frame):
        """
        Compute optical flow and detect panic/chaotic motion.
        Returns motion metrics dict.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        metrics = {
            "avg_magnitude": 0.0,
            "turbulence": 0.0,
            "dominant_direction": "NONE",
            "panic_detected": False,
            "flow_frame": frame.copy()
        }

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_mag = float(np.mean(magnitude))
            turbulence = float(np.std(magnitude))

            self.flow_history.append(avg_mag)

            metrics["avg_magnitude"] = round(avg_mag, 3)
            metrics["turbulence"] = round(turbulence, 3)
            metrics["dominant_direction"] = self._get_dominant_direction(angle, magnitude)
            metrics["panic_detected"] = (
                avg_mag > self.panic_threshold or
                turbulence > self.turbulence_threshold
            )

            # Visualize flow
            metrics["flow_frame"] = self._draw_flow(frame.copy(), flow, magnitude)

        self.prev_gray = gray
        return metrics

    def _get_dominant_direction(self, angle, magnitude):
        """Return dominant crowd movement direction."""
        strong = magnitude > 2.0
        if not np.any(strong):
            return "STILL"
        dominant_angle = float(np.mean(angle[strong]))
        directions = ["RIGHT", "UP-RIGHT", "UP", "UP-LEFT",
                      "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
        idx = int((dominant_angle / (2 * np.pi)) * 8) % 8
        return directions[idx]

    def _draw_flow(self, frame, flow, magnitude, step=16):
        """Draw optical flow vectors."""
        h, w = frame.shape[:2]
        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
        fx = flow[y_coords, x_coords, 0]
        fy = flow[y_coords, x_coords, 1]

        for y, x, dx, dy in zip(y_coords.flat, x_coords.flat, fx.flat, fy.flat):
            if abs(dx) > 1 or abs(dy) > 1:
                end_x = int(x + dx * 3)
                end_y = int(y + dy * 3)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y),
                                (0, 255, 255), 1, tipLength=0.3)
        return frame

    def get_recent_avg_magnitude(self):
        if len(self.flow_history) == 0:
            return 0.0
        return round(float(np.mean(self.flow_history)), 3)