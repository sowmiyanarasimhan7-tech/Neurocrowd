import cv2
import numpy as np
from collections import deque


class MotionAnalyzer:
    """
    Optical-flow-based crowd motion analysis, global + per-zone.

    IMPORTANT PERFORMANCE NOTE: Farneback dense optical flow at full 1080p
    is too slow for real-time on a typical laptop CPU (tested well over a
    second per frame). We compute flow on a downscaled copy of the frame
    (flow_scale, default 0.5 = quarter the pixels) and scale the resulting
    magnitudes back up to stay consistent with the panic/turbulence
    thresholds. This is what keeps the whole pipeline real-time -- if your
    demo machine is fast (or has a GPU), you can raise flow_scale toward
    1.0 for slightly more precise motion detail.
    """

    def __init__(self, history_frames=30, sample_step=16, flow_scale=0.5):
        self.prev_gray = None
        self.flow_history = deque(maxlen=history_frames)
        self.panic_threshold = 8.0        # Flow magnitude threshold (full-res-equivalent units)
        self.turbulence_threshold = 3.5   # Std deviation threshold
        self.sample_step = sample_step
        self.flow_scale = flow_scale

        # cached from the last analyze() call, reused by analyze_zone_motion()
        # so we don't recompute optical flow twice per frame
        self._last_magnitude = None   # in DOWNSCALED coordinate space
        self._last_angle = None

    def analyze(self, frame):
        """
        Compute optical flow and detect panic/chaotic motion (global, whole
        frame). Returns motion metrics dict.
        """
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray_full, None, fx=self.flow_scale, fy=self.flow_scale,
                                 interpolation=cv2.INTER_AREA)
        gray_small = cv2.GaussianBlur(gray_small, (9, 9), 0)

        metrics = {
            "avg_magnitude": 0.0,
            "turbulence": 0.0,
            "dominant_direction": "NONE",
            "panic_detected": False,
            "flow_frame": frame.copy()
        }

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray_small,
                None, 0.5, 2, 12, 2, 5, 1.1, 0
            )

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # scale magnitude back up to be roughly comparable to full-res
            # pixel displacement, since thresholds were tuned in that space
            scale_correction = 1.0 / self.flow_scale
            avg_mag = float(np.mean(magnitude)) * scale_correction
            turbulence = float(np.std(magnitude)) * scale_correction

            self.flow_history.append(avg_mag)

            metrics["avg_magnitude"] = round(avg_mag, 3)
            metrics["turbulence"] = round(turbulence, 3)
            metrics["dominant_direction"] = self._get_dominant_direction(angle, magnitude)
            metrics["panic_detected"] = (
                avg_mag > self.panic_threshold or
                turbulence > self.turbulence_threshold
            )

            metrics["flow_frame"] = self._draw_flow(frame.copy(), flow, magnitude)

            self._last_magnitude = magnitude
            self._last_angle = angle
        else:
            self._last_magnitude = None
            self._last_angle = None

        self.prev_gray = gray_small
        return metrics

    def analyze_zone_motion(self, zones):
        """
        Per-zone motion metrics, reusing the flow field computed in the most
        recent analyze() call. This is what lets the risk engine tell the
        difference between "Zone B is dense but everyone's standing still"
        and "Zone B is dense AND surging" -- the second is the actually
        dangerous case, and a global-only motion score can't distinguish
        them if only one zone out of six is the one surging.

        zones: dict of {zid: {"polygon": np.array(...), ...}} -- pass
        zone_manager.zones directly (polygons at full-frame resolution;
        we scale them down internally to match the flow field).

        Returns: {zid: {"avg_magnitude", "turbulence", "panic_detected"}}
        """
        result = {}

        if self._last_magnitude is None:
            for zid in zones:
                result[zid] = {"avg_magnitude": 0.0, "turbulence": 0.0, "panic_detected": False}
            return result

        h, w = self._last_magnitude.shape
        step = max(4, int(self.sample_step * self.flow_scale))
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        scale_correction = 1.0 / self.flow_scale

        for zid, zone in zones.items():
            polygon_small = (zone["polygon"].astype(np.float32) * self.flow_scale).astype(np.int32)

            zone_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [polygon_small], 1)
            sample_mask = zone_mask[ys, xs].astype(bool)

            sample_mag = self._last_magnitude[ys, xs][sample_mask]

            if sample_mag.size == 0:
                result[zid] = {"avg_magnitude": 0.0, "turbulence": 0.0, "panic_detected": False}
                continue

            avg_mag = float(np.mean(sample_mag)) * scale_correction
            turbulence = float(np.std(sample_mag)) * scale_correction
            panic = avg_mag > self.panic_threshold or turbulence > self.turbulence_threshold

            result[zid] = {
                "avg_magnitude": round(avg_mag, 3),
                "turbulence": round(turbulence, 3),
                "panic_detected": bool(panic),
            }

        return result

    def _get_dominant_direction(self, angle, magnitude):
        """Return dominant crowd movement direction."""
        strong = magnitude > 1.0
        if not np.any(strong):
            return "STILL"
        dominant_angle = float(np.mean(angle[strong]))
        directions = ["RIGHT", "UP-RIGHT", "UP", "UP-LEFT",
                      "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
        idx = int((dominant_angle / (2 * np.pi)) * 8) % 8
        return directions[idx]

    def _draw_flow(self, frame, flow, magnitude, step=10):
        """Draw optical flow vectors, scaled from downscaled flow space
        back up to the full-resolution display frame."""
        h_small, w_small = magnitude.shape
        inv_scale = 1.0 / self.flow_scale

        y_coords, x_coords = np.mgrid[step // 2:h_small:step, step // 2:w_small:step]
        fx = flow[y_coords, x_coords, 0]
        fy = flow[y_coords, x_coords, 1]

        for y, x, dx, dy in zip(y_coords.flat, x_coords.flat, fx.flat, fy.flat):
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                start = (int(x * inv_scale), int(y * inv_scale))
                end = (int((x + dx * 3) * inv_scale), int((y + dy * 3) * inv_scale))
                cv2.arrowedLine(frame, start, end, (0, 255, 255), 1, tipLength=0.3)
        return frame

    def get_recent_avg_magnitude(self):
        if len(self.flow_history) == 0:
            return 0.0
        return round(float(np.mean(self.flow_history)), 3)
