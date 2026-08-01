# NeuroCrowd — Predictive Crowd Safety System

AI-powered crowd monitoring and early-warning system built for the 36-Hour
Hackathon, NIT Trichy — ISRO Bharatiya Antariksh Hackathon track,
"AI for Smart Cities and Transportation."

## The Idea

Most crowd safety systems only show what's already happening — a camera
feed a human has to watch and react to. NeuroCrowd predicts danger a few
seconds *before* it happens, by tracking how fast an area is filling up
and whether crowd motion is calm or surging, then alerting the right
people in time to actually act.

## Problem

Large public gatherings remain vulnerable to overcrowding and panic-induced
stampedes. Existing surveillance is reactive — it records incidents instead
of predicting them, delaying emergency response and increasing casualty risk.

## Solution

NeuroCrowd detects people in live/recorded video, tracks them per named
zone, measures crowd density and motion patterns, and uses a machine
learning model trained to forecast dangerous conditions **before** they
happen — then surfaces that on a live dashboard with actionable
recommendations.

## Architecture
Video source
|
v
Person Detection (YOLOv8, GPU-accelerated, FP16)
|
v
Tracking (persistent IDs, filters single-frame false positives)
|
v
Zone Assignment (density, occupancy %, trend, trend-projected occupancy) <--- Motion Analysis
| (global + per-zone,
v optical flow)
Predictive Risk Model (RandomForest, forecasts ~8 steps ahead)
|
v
Recommendation Engine (actionable alerts per zone)
|
v
Live Dashboard (video + stats panel)


## Project structure

crowd_safety_system/
├── README.md
├── main.py
├── requirements.txt
├── src/
│ ├── detection/person_detector.py
│ ├── tracking/tracker.py
│ ├── analysis/zone_manager.py
│ ├── analysis/motion_analyzer.py
│ ├── prediction/risk_predictor.py
│ ├── recommendation/engine.py
│ ├── dashboard/ui.py
│ └── utils/
├── tools/zone_picker.py
├── zones/configs/venue_config.yaml
├── yolov8n.pt / yolov8s.pt
├── models/risk_model.pkl
│
├── presentation 1.pdf
├── presentation 2.pdf
├── TEAM_INTRODUCTION.pdf
└── HACKATHON_EXPERIENCE.pdf


## What's novel here (say this to judges)

1. **Predictive, not reactive** — trained to forecast risk ~8 steps
   ahead using each zone's occupancy trend, not just the current instant.
2. **Density + motion fusion, per zone** — a dense-but-calm zone and a
   dense-and-surging zone score differently.
3. **Honest crowd estimation** — real detected count + a labeled
   estimated range based on known detector recall limits, plus a
   trend-projected count — every number traces to a real calculation.
4. **Tracking-based flicker correction** — a person only counts once
   confirmed across 2+ consecutive frames.
5. **Dual-channel intent** — dashboard for authorities, designed to
   extend to direct crowd alerts in a full deployment.

## Honesty note for judges

The predictive model trains on **synthetic data** — no real labeled
stampede footage exists to train on in 36 hours. Path to product: pilot
deployment → log real sequences → retrain the same pipeline, zero
architecture changes needed.

## Setup

```bash
cd crowd_safety_system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Before running: define your zones

```bash
python tools/zone_picker.py --source data/videos/crowd_test.mp4
```
Click to outline each zone, `n` to name + set capacity, `s` to save.

## Running

```bash
python main.py --source data/videos/crowd_test.mp4
```
First run takes ~20-30s longer while the risk model trains itself once.

**Controls:** `q` quit, `f` toggle optical-flow view.

## Tech stack

| Component | Technology | Why |
|---|---|---|
| Detection | YOLOv8, FP16 on GPU | fast, GPU-accelerated |
| Tracking | Custom centroid tracker | lightweight |
| Zones | OpenCV polygon geometry | precise per-zone occupancy |
| Motion | OpenCV Farneback optical flow | no training data needed |
| Risk prediction | scikit-learn RandomForest | fast, interpretable |
| Dashboard | OpenCV rendering | real-time |

## Submission contents

- `presentation 1.pdf` / `presentation 2.pdf`
- `TEAM_INTRODUCTION.pdf`
- `HACKATHON_EXPERIENCE.pdf`

## Troubleshooting

- **Low FPS**: check startup log for `Device: CUDA` vs `CPU`.
- **"EST. RANGE 0-0"**: `zone_manager.py` out of date.
- **Video cropped to a corner**: `main.py`/`dashboard/ui.py` out of sync.