# 🛡️ NeuroCrowd: AI-Powered Proactive Crowd Intelligence & Early Risk Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Computer Vision](https://img.shields.io/badge/CV-YOLOv8%20%7C%20OpenCV-green.svg)](https://ultralytics.com/)
[![ML Model](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**NeuroCrowd** is an AI-driven crowd monitoring and predictive risk safety system designed for large public venues (festivals, stadiums, transit hubs, rallies). Existing surveillance systems are primarily **reactive**—detecting incidents only after they occur. NeuroCrowd transforms surveillance into a **proactive decision support system** that measures real physical crowd density ($p/m^2$), analyzes motion panic turbulence, and forecasts stampede risks **3 to 5 minutes in advance**.

---

## 🚀 Key Features

* **🎯 Dense Crowd Head & Person Detection**: Leverages **YOLOv8** with adaptive sensitivity thresholds and contrast-based head contour detection for tightly packed crowds.
* **📐 Spatial Area & Obstacle Calibration Engine**: Calculates true physical density ($p/m^2$) by deducting space occupied by static objects ($A_{\text{usable}} = A_{\text{total}} - A_{\text{obstacle}}$).
* **Fruin's Level of Service (LoS)**: Categorizes density into international safety benchmarks (`SAFE`, `MODERATE`, `HIGH_RISK`, `CRITICAL`).
* **🌪️ Optical Flow Motion & Turbulence**: Tracks crowd flow velocity ($m/s$) and directional chaos index using Farneback Optical Flow.
* **🔮 XGBoost Predictive Risk Classifier**: Forecasts stampede hazards 3 to 5 minutes in advance and computes a continuous **0–100 Danger Index**.
* **📊 Authority Decision Support Dashboard**: Built with **Streamlit**, featuring translucent heatmaps, dynamic zone HUD grids, Plotly danger trend charts, and **Automated Tactical Dispatch Advisories** for emergency responders.
* **📹 Multi-Camera & Video Manager**: Supports live webcam streams, RTSP camera URLs, drag-and-drop MP4 video files, and video folder presets.

---

## 🏗️ System Architecture

```
Live Surveillance Stream ──► YOLOv8 Head Detector ──► Spatial Area Calibration (Net Usable Area)
                                                            │
Tactical Advisories ◄── Streamlit Dashboard ◄── XGBoost 5-Min Risk Model ◄── Farneback Optical Flow
```

---

## 📁 Repository Structure

```
NeuroCrowd/
├── app.py                  # Main Streamlit Dashboard Application
├── cv_engine.py            # OpenCV + YOLOv8 + Optical Flow + Heatmap Engine
├── spatial_calibration.py  # Net Usable Space & Fruin Density Calibration Engine
├── predictive_model.py     # XGBoost Risk Classifier & Danger Index Forecaster
├── config.py               # Fruin benchmarks, risk theme colors & tactical advisories
├── train_model.py          # Model training & retraining script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/sowmiyanarasimhan7-tech/Neurocrowd.git
cd Neurocrowd
```

### 2. Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the NeuroCrowd Application
```powershell
streamlit run app.py
```

Access the interactive dashboard in your web browser at `http://localhost:8501`.

---

## 🎓 Academic Credit & Author Information

Developed for **AM5305 – Machine Learning PBL Project Review**.

* **Institution**: Chennai Institute of Technology, Chennai (Autonomous)
* **Author**: Sowmiya Narasimhan & Team
* **Tech Stack**: Python, OpenCV, YOLOv8, XGBoost, Streamlit, Plotly