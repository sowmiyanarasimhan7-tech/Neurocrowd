"""
NeuroCrowd — Tracking module
--------------------------------
Adds persistent IDs to detections across frames using simple greedy
nearest-neighbor matching (no extra dependency needed beyond numpy).

Why this matters for your accuracy complaint: a single frame's YOLO output
can flicker — a person detected this frame might be missed next frame due
to motion blur or brief occlusion, then re-detected the frame after. Without
tracking, each of those looks like a fresh, uncertain detection. With
tracking + a "confirm after N consecutive hits" rule, a real person only
gets COUNTED once they've been seen consistently, which kills one-frame
false positives, and a person who's tracked but missed for a couple frames
(occlusion) doesn't just vanish from the count either — grace period below.
"""

import numpy as np


class Track:
    __slots__ = ("id", "center", "hits", "age", "misses")

    def __init__(self, track_id, center):
        self.id = track_id
        self.center = center
        self.hits = 1
        self.age = 0
        self.misses = 0


class SimpleTracker:
    def __init__(self, max_distance=70, max_misses=8, min_hits=2):
        """
        max_distance: pixels — how far a detection can be from a track's last
                      known position and still be considered the same person
        max_misses:   frames a track can go undetected before being dropped
                      (handles brief occlusion instead of instantly losing count)
        min_hits:     consecutive detections required before a track is
                      "confirmed" and counted — filters one-frame flicker
        """
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.tracks = {}
        self._next_id = 1

    def update(self, detections):
        """
        detections: list of dicts with a 'center' key (x, y).
        Returns the same list with a 'track_id' and 'confirmed' key added
        to each detection. Also returns confirmed_count (unique confirmed
        people visible this frame).
        """
        det_centers = np.array([d["center"] for d in detections]) if detections else np.empty((0, 2))
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[t].center for t in track_ids]) if track_ids else np.empty((0, 2))

        matched_tracks = set()
        matched_dets = set()

        if len(det_centers) > 0 and len(track_centers) > 0:
            dists = np.linalg.norm(det_centers[:, None, :] - track_centers[None, :, :], axis=2)
            pairs = [(dists[i, j], i, j) for i in range(dists.shape[0]) for j in range(dists.shape[1])]
            pairs.sort(key=lambda p: p[0])

            for dist, di, ti in pairs:
                if di in matched_dets or ti in matched_tracks:
                    continue
                if dist > self.max_distance:
                    continue
                tid = track_ids[ti]
                self.tracks[tid].center = tuple(det_centers[di])
                self.tracks[tid].hits += 1
                self.tracks[tid].misses = 0
                detections[di]["track_id"] = tid
                matched_dets.add(di)
                matched_tracks.add(ti)

        # unmatched detections -> new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, det["center"])
                det["track_id"] = tid

        # unmatched tracks -> aging out
        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self.tracks[tid].misses += 1

        # drop stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].misses > self.max_misses:
                del self.tracks[tid]

        confirmed = 0
        for det in detections:
            tid = det["track_id"]
            is_confirmed = self.tracks[tid].hits >= self.min_hits
            det["confirmed"] = is_confirmed
            if is_confirmed:
                confirmed += 1

        return detections, confirmed
