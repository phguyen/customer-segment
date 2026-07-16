from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# =========================================================
# ĐƯỜNG DẪN
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "rfm_scaler.pkl"
MAPPING_PATH = MODEL_DIR / "persona_mapping.json"


# =========================================================
# TẢI MODEL
# =========================================================
def load_prediction_assets():
    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        MAPPING_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Không tìm thấy các file model sau:\n"
            + "\n".join(missing_files)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        persona_mapping = json.load(file)

    return model, scaler, persona_mapping


# =========================================================
# DỰ ĐOÁN KHÁCH HÀNG
# =========================================================
def predict_customer(
    recency: float,
    frequency: float,
    monetary: float,
) -> dict:
    if recency < 0:
        raise ValueError("Recency không được nhỏ hơn 0.")

    if frequency <= 0:
        raise ValueError("Frequency phải lớn hơn 0.")

    if monetary <= 0:
        raise ValueError("Monetary phải lớn hơn 0.")

    model, scaler, persona_mapping = load_prediction_assets()

    # Giữ đúng tên và thứ tự cột như lúc huấn luyện
    rfm_input = pd.DataFrame(
        [
            {
                "Recency": float(recency),
                "Frequency": float(frequency),
                "Monetary": float(monetary),
            }
        ],
        columns=[
            "Recency",
            "Frequency",
            "Monetary",
        ],
    )

    # Quan trọng: notebook đã dùng log1p trước StandardScaler
    rfm_log = np.log1p(rfm_input)

    # Dùng scaler đã fit, tuyệt đối không fit lại
    rfm_scaled = scaler.transform(rfm_log)

    # Dự đoán bằng K-Means đã huấn luyện
    cluster = int(model.predict(rfm_scaled)[0])

    cluster_name = persona_mapping.get(
        str(cluster),
        persona_mapping.get(
            cluster,
            f"Cluster {cluster}",
        ),
    )

    # Trường hợp JSON lưu object thay vì chuỗi
    # if isinstance(cluster_name, dict):
    #     cluster_name = (
    #         cluster_name.get("name")
    #         or cluster_name.get("ClusterName")
    #         or cluster_name.get("label")
    #         or f"Cluster {cluster}"
    #     )
    if isinstance(cluster_name, dict):
        cluster_description = cluster_name.get(
            "description",
            "",
        )

        cluster_name = (
            cluster_name.get("persona")
            or cluster_name.get("name")
            or cluster_name.get("ClusterName")
            or cluster_name.get("label")
            or f"Cluster {cluster}"
        )
    else:
        cluster_description = ""

    return {
            "Cluster": cluster,
            "ClusterName": str(cluster_name),
            "ClusterDescription": str(cluster_description),
            "Recency": float(recency),
            "Frequency": float(frequency),
            "Monetary": float(monetary),
            "ModelName": type(model).__name__,
            "ScalerName": type(scaler).__name__,
            "ModelPath": str(MODEL_PATH),
        }