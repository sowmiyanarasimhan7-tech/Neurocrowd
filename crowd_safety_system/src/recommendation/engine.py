import yaml
from datetime import datetime

class RecommendationEngine:
    def __init__(self, config_path="zones/configs/venue_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.log = []

    def generate(self, zone_summary, motion_metrics, audio_panic=False, overall_risk="SAFE"):
        """
        Generate actionable recommendations based on current situation.
        Returns list of recommendation strings.
        """
        recs = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        for zid, zone in zone_summary.items():
            density = zone["density"]
            risk = zone["risk"]
            count = zone["count"]
            exits = self.config["zones"][self._zone_index(zid)]["exits"]

            if risk == "CRITICAL":
                recs.append(f"🔴 [{timestamp}] CRITICAL — Zone {zid} ({zone['name']}): {count} people. IMMEDIATELY open {exits[0]}.")
                recs.append(f"🔴 [{timestamp}] Deploy crowd-control personnel near Zone {zid}.")
                recs.append(f"🔴 [{timestamp}] Halt all NEW entries into Zone {zid}.")

            elif risk == "WARNING":
                recs.append(f"🟡 [{timestamp}] WARNING — Zone {zid} at {int(density*100)}% capacity. Consider opening {exits[0]}.")
                if len(exits) > 1:
                    recs.append(f"🟡 [{timestamp}] Redirect crowd flow toward {exits[1]} from Zone {zid}.")

        if motion_metrics.get("panic_detected"):
            recs.append(f"🔴 [{timestamp}] PANIC MOTION DETECTED — Turbulence: {motion_metrics['turbulence']}. Activate emergency protocol.")
            recs.append(f"🔴 [{timestamp}] Announce calm via PA system immediately.")

        if audio_panic:
            recs.append(f"🔴 [{timestamp}] AUDIO PANIC DETECTED — Abnormal crowd sound pattern. Verify emergency on ground.")

        if overall_risk == "SAFE" and not recs:
            recs.append(f"✅ [{timestamp}] All zones SAFE. Maintain regular monitoring.")

        self.log.extend(recs)
        return recs

    def _zone_index(self, zid):
        for i, z in enumerate(self.config["zones"]):
            if z["id"] == zid:
                return i
        return 0

    def get_log(self, last_n=20):
        return self.log[-last_n:]