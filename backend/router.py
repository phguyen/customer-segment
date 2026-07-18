import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json as _json

# Import các hàm tương tác Cơ sở dữ liệu từ thư mục database
from backend.database.connection import log_to_db, get_prediction_history_from_db

# Khởi tạo APIRouter để nhúng vào main.py
router = APIRouter()

# -------------------------------------------------------------------------
# 1. TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA LÊN BỘ NHỚ (LOAD ONCE)
# -------------------------------------------------------------------------
MODEL_PATH = "models/kmeans_model_v2.pkl"
SCALER_PATH = "models/rfm_scaler.pkl"
PERSONA_JSON_PATH = "models/persona_mapping.json"  
THRESHOLD_JSON_PATH = "models/cluster_distance_thresholds.json"  

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[Backend] Đã tải mô hình K-Means và Bộ chuẩn hóa thành công.")
else:
    model, scaler = None, None
    print("[Cảnh báo] Chưa tìm thấy file mô hình hoặc bộ chuẩn hóa trong thư mục models/.")

NOW = datetime.now()

persona_map = {}
if os.path.exists(PERSONA_JSON_PATH):
    try:
        with open(PERSONA_JSON_PATH, "r", encoding="utf-8") as f:
            persona_map = _json.load(f)
        print("[Backend] Đã nạp file cấu hình Persona JSON thành công.")
    except Exception as e:
        print(f"[Lỗi] Không thể đọc file JSON Persona: {e}")

# Nạp ngưỡng khoảng cách theo từng cụm để đánh giá độ tin cậy
cluster_thresholds = {}
if os.path.exists(THRESHOLD_JSON_PATH):
    try:
        with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
            cluster_thresholds = _json.load(f)
        print("[Backend] Đã nạp ngưỡng khoảng cách theo cụm thành công.")
    except Exception as e:
        print(f"[Lỗi] Không thể đọc file JSON ngưỡng khoảng cách: {e}")

# -------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA ĐỊNH DẠNG DỮ LIỆU ĐẦU VÀO (DATA VALIDATION)
# -------------------------------------------------------------------------
class SinglePredictRequest(BaseModel):
    last_purchase_date: str = Field(..., description="Ngày mua hàng cuối cùng (YYYY-MM-DD)", example="2011-12-01")
    frequency: float = Field(..., description="Tổng số hóa đơn mua hàng", example=5.0)
    monetary: float = Field(..., description="Tổng số tiền chi tiêu tích lũy", example=2500000.0)

# -------------------------------------------------------------------------
# 3. ĐỊNH TUYẾN CÁC ENDPOINT (ROUTES)
# -------------------------------------------------------------------------

@router.get("/health")
def health_check():
    if model is None or scaler is None:
        return {"status": "unhealthy", "message": "Mô hình hoặc bộ chuẩn hóa chưa được tải thành công."}
    return {"status": "healthy"}

@router.get("/model-info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng.")
    return {
        "Algorithm": "K-Means",
        "Số cụm": int(model.n_clusters),
        "Chỉ số Silhouette": round(getattr(model, "silhouette_score_f_", 0.3200), 4),  
        "Davies-Bouldin Index": round(getattr(model, "davies_bouldin_index_f_", 1.0172), 4)
    }

@router.post("/predict")
def predict_customer_segment(data: SinglePredictRequest):
    """
    Endpoint 3: Nhận thông tin thô, tiền xử lý, lưu DB, dự đoán và trả về độ tin cậy.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng, thiếu file mô hình.")

    try:
        try:
            user_date = datetime.strptime(data.last_purchase_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Định dạng ngày không hợp lệ. Vui lòng dùng YYYY-MM-DD.")

        recency_days = (NOW - user_date).days
        if recency_days < 0:
            raise HTTPException(status_code=400, detail="Ngày mua hàng cuối cùng không thể ở tương lai.")

        # Tiền xử lý dữ liệu đơn lẻ
        raw_features = pd.DataFrame(
            [[float(recency_days), data.frequency, data.monetary]],
            columns=['Recency', 'Frequency', 'Monetary']
        )
        log_features = raw_features.apply(np.log1p)
        scaled_features = scaler.transform(log_features)

        # Dự đoán Cluster ID
        cluster_id = int(model.predict(scaled_features)[0])

        # ĐÁNH GIÁ ĐỘ TIN CẬY (CONFIDENCE) DỰA TRÊN KHOẢNG CÁCH
        distances_to_all_centers = model.transform(scaled_features)[0]
        distance_to_assigned_center = float(distances_to_all_centers[cluster_id])

        threshold = cluster_thresholds.get(str(cluster_id))
        if threshold is not None:
            # Nếu vượt quá ngưỡng khoảng cách an toàn, độ tin cậy sẽ thấp (lo ngại điểm biên lệch)
            confidence = "low" if distance_to_assigned_center > threshold else "high"
        else:
            confidence = "unknown"

        # Ánh xạ thành nhãn Persona kinh doanh
        key = str(cluster_id)
        if key in persona_map:
            persona_info = persona_map[key]
        else:
            persona_info = {"persona": f"Nhóm khách hàng {cluster_id}", "description": "N/A"}

        persona_name = persona_info["persona"]
        description = persona_info.get("description", "")

        # Lưu lịch sử dự đoán
        log_to_db(
            recency=float(recency_days),
            frequency=data.frequency,
            monetary=data.monetary,
            cluster_id=cluster_id,
            cluster_label=persona_name
        )

        return {
            "cluster_id": cluster_id,
            "persona": persona_name,
            "description": description,
            "confidence": confidence,
            "distance_to_center": round(distance_to_assigned_center, 4)
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

@router.get("/history")
def get_prediction_history():
    try:
        history_rows = get_prediction_history_from_db()
        return {"history": history_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi truy xuất lịch sử: {str(e)}")