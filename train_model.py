"""
NeuroCrowd Model Training Script
"""

import os
import argparse
from ultralytics import YOLO
from predictive_model import CrowdRiskPredictor

def train_yolov8_crowd(dataset_yaml="crowdhuman.yaml", epochs=20, batch=16, imgsz=640):
    print(f"🚀 Starting YOLOv8 Training on Dataset: {dataset_yaml}")
    model = YOLO("yolov8n.pt")
    results = model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch, name="neurocrowd_head_yolo")
    print("✅ YOLOv8 Crowd Head Model Training Complete!")
    return results

def retrain_xgboost_risk_model():
    print("🚀 Retraining XGBoost Crowd Risk Forecast Model...")
    predictor = CrowdRiskPredictor()
    print("✅ XGBoost Model Successfully Retrained!")
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["yolo", "xgboost", "all"], default="xgboost")
    parser.add_argument("--data", type=str, default="crowdhuman.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    if args.mode in ["xgboost", "all"]:
        retrain_xgboost_risk_model()
    if args.mode in ["yolo", "all"]:
        if os.path.exists(args.data):
            train_yolov8_crowd(dataset_yaml=args.data, epochs=args.epochs)
        else:
            print(f"⚠️ Dataset YAML '{args.data}' not found.")