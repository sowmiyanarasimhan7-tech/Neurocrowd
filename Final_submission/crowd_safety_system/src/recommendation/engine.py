import time
import yaml
from datetime import datetime


class RecommendationEngine:
    """
    Generates actionable recommendations from the current zone/motion state.

    Two behaviors, kept intentionally separate:
      - generate() returns the FULL current set of recommendations every
        call, so your live dashboard always reflects the current situation
        (this is what main.py should display every frame).
      - the persistent self.log (get_log()) is cooldown-gated per
        zone+risk-level so it doesn't fill up with the same "Zone B
        CRITICAL" message repeated 30 times a second -- this is what you'd
        show as an audit trail / incident history if you add that to the
        dashboard, and it's a legitimate thing to point to when judges ask
        "how would authorities review what happened afterward".
    """

    def __init__(self, config_path="zones/configs/venue_config.yaml", log_cooldown_seconds=8):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.log = []
        self.log_cooldown_seconds = log_cooldown_seconds
        self._last_logged = {}

    def _should_log(self, key):
        now = time.time()
        last = self._last_logged.get(key, 0)
        if now - last >= self.log_cooldown_seconds:
            self._last_logged[key] = now
            return True
        return False

    def generate(self, zone_summary, motion_metrics, audio_panic=False, overall_risk="SAFE"):
        """
        Generate actionable recommendations based on current situation.
        Returns list of recommendation strings (current state, every call).
        """
        recs = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        for zid, zone in zone_summary.items():
            density = zone["density"]
            risk = zone["risk"]
            count = zone["count"]
            trend = zone.get("trend_arrow", "-")
            exits = self.config["zones"][self._zone_index(zid)]["exits"]

            if risk == "CRITICAL":
                msg1 = f"[{timestamp}] CRITICAL - Zone {zid} ({zone['name']}): {count} people. IMMEDIATELY open {exits[0]}."
                msg2 = f"[{timestamp}] Deploy crowd-control personnel near Zone {zid}."
                msg3 = f"[{timestamp}] Halt all NEW entries into Zone {zid}."
                recs += [msg1, msg2, msg3]
                if self._should_log(f"{zid}-CRITICAL"):
                    self.log.extend([msg1, msg2, msg3])

            elif risk == "WARNING":
                trend_note = " and rising" if trend == "UP" else ""
                msg1 = f"[{timestamp}] WARNING - Zone {zid} at {int(density*100)}% capacity{trend_note}. Consider opening {exits[0]}."
                recs.append(msg1)
                if len(exits) > 1:
                    msg2 = f"[{timestamp}] Redirect crowd flow toward {exits[1]} from Zone {zid}."
                    recs.append(msg2)
                    if self._should_log(f"{zid}-WARNING"):
                        self.log.extend([msg1, msg2])
                elif self._should_log(f"{zid}-WARNING"):
                    self.log.append(msg1)

            elif trend == "UP" and risk == "SAFE":
                # early heads-up: not dangerous yet, but the predictive
                # signal (trend) says it's worth a mention -- this is the
                # kind of line that demonstrates the "before it happens"
                # pitch in a live demo
                msg = f"[{timestamp}] Zone {zid} occupancy trending upward -- monitor."
                recs.append(msg)

        if motion_metrics.get("panic_detected"):
            msg1 = f"[{timestamp}] PANIC MOTION DETECTED - Turbulence: {motion_metrics['turbulence']}. Activate emergency protocol."
            msg2 = f"[{timestamp}] Announce calm via PA system immediately."
            recs += [msg1, msg2]
            if self._should_log("GLOBAL-PANIC"):
                self.log.extend([msg1, msg2])

        if audio_panic:
            msg = f"[{timestamp}] AUDIO PANIC DETECTED - Abnormal crowd sound pattern. Verify emergency on ground."
            recs.append(msg)
            if self._should_log("GLOBAL-AUDIO"):
                self.log.append(msg)

        if overall_risk == "SAFE" and not recs:
            recs.append(f"[{timestamp}] All zones SAFE. Maintain regular monitoring.")

        return recs

    def _zone_index(self, zid):
        for i, z in enumerate(self.config["zones"]):
            if z["id"] == zid:
                return i
        return 0

    def get_log(self, last_n=20):
        return self.log[-last_n:]
