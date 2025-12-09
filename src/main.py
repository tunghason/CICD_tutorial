# src/train_model.py

import os
import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn
from ultralytics import YOLO
from ultralytics import settings

from dotenv import load_dotenv
load_dotenv()

# Update a setting
settings.update({"mlflow": True})

# # --- 1. Cấu hình Tracking MLflow ---
# # Lấy URI từ biến môi trường đã set trong setup_remote.sh
# mlflow.set_tracking_uri("sqlite:///mlruns.db") 
# mlflow.set_experiment("Machine_1_Training")

DATA_FILE_PATH = "/media/HDD0/tunghs4/Uchiyama/CICD_tutorial/data/Machine1"
# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

def on_train_end(trainer):
    mlflow.pytorch.log_model(trainer.model, name="model")

def get_git_commit():
    """Lấy Git Commit ID hiện tại."""
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "UNKNOWN_COMMIT"

def train_and_log():
    # Tạo MLflow Run mới
    # with mlflow.start_run(run_name=f"Yolo_cls_machine_1") as run:
    #     print("Bắt đầu Huấn luyện và Ghi lại MLflow...")
    #     mlflow_artifact_path = mlflow.get_artifact_uri()
    #     # --- 2. Ghi lại phiên bản (Reproducibility) ---
    #     git_commit = get_git_commit()
        
    #     mlflow.log_param("git_commit_id", git_commit)
    #     mlflow.log_param("model_type", "Yolo11n-cls")
        
       # Train the model    
    model.add_callback("on_train_end", on_train_end)
    results = model.train(data=DATA_FILE_PATH, epochs=5, imgsz=640)

    
        # # --- 6. Lưu Mô hình (DVC và MLflow Artifact) ---
        # # Lưu mô hình vào thư mục models/
        # os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
        # joblib.dump(model, MODEL_FILE_PATH)
        
        # # DVC add mô hình để version hóa
        # subprocess.run(['dvc', 'add', MODEL_FILE_PATH], check=True)
        # print(f"Đã DVC add mô hình mới: {MODEL_FILE_PATH}.dvc")

        # # Ghi lại DVC pointer (.dvc file) vào MLflow Artifact
        # mlflow.log_artifact(f"{MODEL_FILE_PATH}.dvc")

        # # Ghi lại mô hình vào MLflow Registry
        # mlflow.sklearn.log_model(
        #     sk_model=model, 
        #     artifact_path="model",
        #     registered_model_name="Production_Model_Machine_A"
        # )
        # print("Đã đăng ký mô hình vào MLflow Registry.")

if __name__ == "__main__":
    train_and_log()