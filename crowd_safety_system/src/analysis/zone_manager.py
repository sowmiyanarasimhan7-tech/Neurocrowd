import cv2
import yaml
import numpy as np
from collections import deque


class ZoneManager:

    def __init__(self,
                 config_path="zones/configs/venue_config.yaml",
                 history_len=30):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_width = self.config["frame_width"]
        self.base_height = self.config["frame_height"]

        self.zones = {}

        for z in self.config["zones"]:

            self.zones[z["id"]] = {

                "name": z["name"],

                "polygon_original": np.array(
                    z["polygon"],
                    dtype=np.float32
                ),

                "polygon": np.array(
                    z["polygon"],
                    dtype=np.int32
                ),

                "max_capacity": z["max_capacity"],

                "current_count": 0,

                "density": 0,

                "occupancy": 0,

                "risk_level": "SAFE",

                "exits": z["exits"],

                # rolling history of occupancy % -> used for trend + the
                # predictive risk model's "is this getting worse" features
                "history": deque(maxlen=history_len),
            }

    def update_scale(self, frame):

        h, w = frame.shape[:2]

        sx = w / self.base_width
        sy = h / self.base_height

        for zone in self.zones.values():

            pts = zone["polygon_original"].copy()

            pts[:, 0] *= sx
            pts[:, 1] *= sy

            zone["polygon"] = pts.astype(np.int32)

    def assign_person_to_zone(self, center):

        x, y = center

        for zid, zone in self.zones.items():

            inside = cv2.pointPolygonTest(
                zone["polygon"],
                (float(x), float(y)),
                False
            )

            if inside >= 0:
                return zid

        return None

    def count_people_per_zone(self,
                              detections,
                              frame=None,
                              confirmed_only=True):
        """
        confirmed_only: if detections have gone through the tracker (have a
        'confirmed' key), only count confirmed ones. This is what actually
        fixes flicker-driven inaccurate counts — unconfirmed (single-frame)
        detections are shown dimmed on screen but excluded from the zone
        tallies that drive risk scoring.
        """

        if frame is not None:
            self.update_scale(frame)

        for z in self.zones.values():
            z["current_count"] = 0

        zone_people = {}

        for det in detections:

            if confirmed_only and not det.get("confirmed", True):
                continue

            zid = self.assign_person_to_zone(det["center"])

            if zid is None:
                continue

            self.zones[zid]["current_count"] += 1

            zone_people.setdefault(zid, []).append(det)

        self.calculate_density()

        return zone_people

    def calculate_density(self):

        for zone in self.zones.values():

            count = zone["current_count"]

            cap = zone["max_capacity"]

            occupancy = count / cap

            zone["density"] = occupancy

            zone["occupancy"] = occupancy * 100

            zone["history"].append(zone["occupancy"])

            if occupancy < 0.50:

                zone["risk_level"] = "SAFE"

            elif occupancy < 0.80:

                zone["risk_level"] = "WARNING"

            else:

                zone["risk_level"] = "CRITICAL"

    def get_zone_trend(self, zid):
        """
        Returns (slope, arrow) describing whether this zone's occupancy is
        climbing, falling, or flat over its recent history. slope is in
        percentage-points per frame. This is the raw signal the predictive
        risk model uses to forecast danger before the threshold is crossed.
        """
        hist = self.zones[zid]["history"]
        if len(hist) < 4:
            return 0.0, "-"
        y = np.array(hist)
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0])
        if slope > 0.4:
            arrow = "UP"
        elif slope < -0.4:
            arrow = "DOWN"
        else:
            arrow = "FLAT"
        return slope, arrow

    def draw_zones(self,
                   frame):

        colors = {

            "SAFE": (0, 200, 0),

            "WARNING": (0, 180, 255),

            "CRITICAL": (0, 0, 255)

        }

        overlay = frame.copy()

        for zid, zone in self.zones.items():

            color = colors[zone["risk_level"]]

            cv2.fillPoly(
                overlay,
                [zone["polygon"]],
                color
            )

        cv2.addWeighted(
            overlay,
            0.20,
            frame,
            0.80,
            0,
            frame
        )

        for zid, zone in self.zones.items():

            color = colors[zone["risk_level"]]

            cv2.polylines(
                frame,
                [zone["polygon"]],
                True,
                color,
                2
            )

            M = cv2.moments(zone["polygon"])

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])

            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                frame,
                f"{zid}",
                (cx - 15, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"{zone['current_count']} / {zone['max_capacity']}",
                (cx-45,cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"{zone['occupancy']:.1f}%",
                (cx-30,cy+22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

        return frame

    def get_zone_summary(self):

        summary = {}

        for zid, zone in self.zones.items():

            slope, arrow = self.get_zone_trend(zid)

            summary[zid] = {

                "name": zone["name"],

                "count": zone["current_count"],

                "density": round(zone["density"],3),

                "occupancy": round(zone["occupancy"],1),

                "risk": zone["risk_level"],

                "capacity": zone["max_capacity"],

                "exits": zone["exits"],

                "trend_slope": round(slope, 3),

                "trend_arrow": arrow,

            }

        return summary

    def get_total_people(self):

        return sum(
            z["current_count"]
            for z in self.zones.values()
        )
