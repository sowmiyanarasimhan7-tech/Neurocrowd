"""
NeuroCrowd - Dashboard UI module
------------------------------------
Video stays clean and fully visible on the left; all stats, risk badges,
per-zone occupancy bars + trend arrows, motion/panic status, a live danger
sparkline, and the recommendation feed live in a dark panel on the right.

The sparkline is worth pointing to during your demo: because the risk
model outputs a continuous danger index (not just a label), you can watch
it climb *before* a zone actually turns red -- that's the concrete visual
proof of "predictive, not reactive" you can point to live on stage.
"""

import cv2
import numpy as np

BG_PANEL      = (26, 22, 18)
BG_HEADER     = (46, 36, 20)
TEXT_MAIN     = (240, 240, 240)
TEXT_DIM      = (160, 160, 160)
ACCENT        = (255, 190, 60)
RISK_COLORS = {
    "SAFE":     (90, 210, 90),
    "WARNING":  (0, 180, 255),
    "CRITICAL": (50, 50, 255),
}
TREND_GLYPH = {"UP": "^ rising", "DOWN": "v falling", "FLAT": "- steady", "-": ""}


def _rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, thickness)


def _draw_progress_bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 50, 45), -1, cv2.LINE_AA)
    fill_w = int(w * min(pct, 100) / 100)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 85, 80), 1, cv2.LINE_AA)


def _put_text(img, text, org, scale=0.5, color=TEXT_MAIN, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _draw_sparkline(img, x, y, w, h, values, color=ACCENT, max_value=100.0):
    """values: iterable of floats in [0, max_value]. Draws a filled area
    sparkline in the given box, oldest->newest left-to-right."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (36, 32, 28), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 65, 58), 1, cv2.LINE_AA)
    vals = list(values)
    if len(vals) < 2:
        return
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i / (n - 1) * w)
        py = y + h - int(min(max(v, 0), max_value) / max_value * h)
        pts.append((px, py))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
    # threshold guide lines at 50 / 80 (warning / critical)
    for frac, c in [(0.5, RISK_COLORS["WARNING"]), (0.8, RISK_COLORS["CRITICAL"])]:
        gy = y + h - int(frac * h)
        cv2.line(img, (x, gy), (x + w, gy), c, 1, cv2.LINE_AA)


def draw_sidebar(height, width, zone_summary, overall_risk, proba, motion_metrics,
                  zone_motion, recs, total_people, fps, risk_history):
    panel = np.full((height, width, 3), BG_PANEL, dtype=np.uint8)
    pad = 16
    y = 0

    # ---- header --------------------------------------------------------
    cv2.rectangle(panel, (0, 0), (width, 68), BG_HEADER, -1)
    _put_text(panel, "NEUROCROWD", (pad, 30), 0.72, (255, 255, 255), 2)
    _put_text(panel, "Predictive Crowd Safety System", (pad, 52), 0.4, TEXT_DIM, 1)
    y = 82

    # ---- overall risk badge --------------------------------------------
    risk_color = RISK_COLORS.get(overall_risk, TEXT_MAIN)
    _rounded_rect(panel, (pad, y), (width - pad, y + 58), (40, 38, 34), radius=9)
    cv2.rectangle(panel, (pad, y), (pad + 7, y + 58), risk_color, -1)
    _put_text(panel, "OVERALL RISK (forecast)", (pad + 18, y + 20), 0.38, TEXT_DIM, 1)
    _put_text(panel, overall_risk, (pad + 18, y + 47), 0.78, risk_color, 2)
    if proba:
        _put_text(panel, f"conf {int(max(proba.values()) * 100)}%", (width - 120, y + 36), 0.45, TEXT_DIM, 1)
    y += 72

    # ---- risk trend sparkline -------------------------------------------
    _put_text(panel, "DANGER INDEX (last ~30s)", (pad, y + 12), 0.4, ACCENT, 1)
    y += 18
    _draw_sparkline(panel, pad, y, width - 2 * pad, 42, risk_history, color=(255, 220, 120))
    y += 54

    # ---- total people / fps ----------------------------------------------
    half = (width - 2 * pad - 10) // 2
    _rounded_rect(panel, (pad, y), (pad + half, y + 50), (40, 38, 34), radius=8)
    _put_text(panel, "TOTAL PEOPLE", (pad + 10, y + 18), 0.35, TEXT_DIM, 1)
    _put_text(panel, str(total_people), (pad + 10, y + 40), 0.7, (255, 255, 255), 2)

    _rounded_rect(panel, (pad + half + 10, y), (width - pad, y + 50), (40, 38, 34), radius=8)
    _put_text(panel, "FPS", (pad + half + 20, y + 18), 0.35, TEXT_DIM, 1)
    _put_text(panel, f"{fps:.1f}", (pad + half + 20, y + 40), 0.7, (255, 255, 255), 2)
    y += 64

    # ---- zones ------------------------------------------------------------
    _put_text(panel, "ZONE OCCUPANCY", (pad, y + 12), 0.42, ACCENT, 1)
    y += 22
    cv2.line(panel, (pad, y), (width - pad, y), (60, 56, 50), 1)
    y += 12

    zone_motion = zone_motion or {}
    for zid, z in zone_summary.items():
        color = RISK_COLORS.get(z["risk"], TEXT_MAIN)
        trend = TREND_GLYPH.get(z.get("trend_arrow", "-"), "")
        panic = zone_motion.get(zid, {}).get("panic_detected", False)

        _put_text(panel, f'{zid} - {z["name"]}', (pad, y + 11), 0.42, TEXT_MAIN, 1)
        _put_text(panel, f'{z["count"]}/{z["capacity"]}', (width - pad - 68, y + 11), 0.38, TEXT_DIM, 1)
        y += 16
        _draw_progress_bar(panel, pad, y, width - 2 * pad - 50, 12, z["occupancy"], color)
        _put_text(panel, f'{z["occupancy"]:.0f}%', (width - pad - 42, y + 11), 0.38, color, 1)
        y += 18
        status_line = trend
        if panic:
            status_line = (status_line + "  PANIC MOTION").strip()
        if status_line:
            sc = RISK_COLORS["CRITICAL"] if panic else TEXT_DIM
            _put_text(panel, status_line, (pad, y + 10), 0.34, sc, 1)
        y += 16

    y += 6
    cv2.line(panel, (pad, y), (width - pad, y), (60, 56, 50), 1)
    y += 14

    # ---- global motion -------------------------------------------------
    _put_text(panel, "GLOBAL MOTION", (pad, y + 12), 0.42, ACCENT, 1)
    y += 22
    _put_text(panel, f'Magnitude {motion_metrics.get("avg_magnitude", 0.0):.2f}   '
                      f'Turbulence {motion_metrics.get("turbulence", 0.0):.2f}   '
                      f'Dir {motion_metrics.get("dominant_direction", "-")}',
              (pad, y + 11), 0.36, TEXT_MAIN, 1)
    y += 24

    cv2.line(panel, (pad, y), (width - pad, y), (60, 56, 50), 1)
    y += 14

    # ---- recommendations (fills the rest) --------------------------------
    _put_text(panel, "LIVE RECOMMENDATIONS", (pad, y + 12), 0.42, ACCENT, 1)
    y += 24
    max_lines = max(0, (height - y - pad) // 20)
    for rec in recs[:max_lines]:
        clean = rec.strip(" -")
        if len(clean) > 50:
            clean = clean[:50] + "..."
        color = RISK_COLORS["CRITICAL"] if "CRITICAL" in rec or "PANIC" in rec else \
                RISK_COLORS["WARNING"] if "WARNING" in rec else RISK_COLORS["SAFE"]
        _put_text(panel, clean, (pad, y + 10), 0.34, color, 1)
        y += 18

    return panel


def letterbox(frame, target_w, target_h):
    """Resize frame to fit within target_w x target_h preserving aspect
    ratio, padding with black bars so the FULL frame is always visible."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def compose_display(video_frame, sidebar_width, video_target_w, video_target_h,
                     zone_summary, overall_risk, proba, motion_metrics, zone_motion,
                     recs, total_people, fps, risk_history):
    video_area = letterbox(video_frame, video_target_w, video_target_h)
    sidebar = draw_sidebar(video_target_h, sidebar_width, zone_summary, overall_risk,
                            proba, motion_metrics, zone_motion, recs, total_people, fps, risk_history)
    return np.hstack([video_area, sidebar])
