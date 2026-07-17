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
MODEL_PATH = "models/kmeans_model.pkl"
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
# Tự động đọc cấu hình Persona từ file JSON lên bộ nhớ khi khởi động server
if os.path.exists(PERSONA_JSON_PATH):
    try:
        with open(PERSONA_JSON_PATH, "r", encoding="utf-8") as f:
            persona_map = _json.load(f)
        print("[Backend] Đã nạp file cấu hình Persona JSON thành công.")
    except Exception as e:
        print(f"[Lỗi] Không thể đọc file JSON Persona: {e}")
else:
    print(f"[Cảnh báo] Không tìm thấy file JSON cấu hình Persona tại: {PERSONA_JSON_PATH}")

# Nạp ngưỡng khoảng cách theo từng cụm (dùng để phát hiện khách hàng bất thường/ngoại lệ)
cluster_thresholds = {}
if os.path.exists(THRESHOLD_JSON_PATH):
    try:
        with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
            cluster_thresholds = _json.load(f)
        print("[Backend] Đã nạp ngưỡng khoảng cách theo cụm thành công.")
    except Exception as e:
        print(f"[Lỗi] Không thể đọc file JSON ngưỡng khoảng cách: {e}")
else:
    print(f"[Cảnh báo] Không tìm thấy file ngưỡng khoảng cách tại: {THRESHOLD_JSON_PATH} "
          f"(chạy lại train_final.py bản mới để tạo file này).")

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
    """
    Endpoint 1: Kiểm tra trạng thái hoạt động của server và tệp .pkl
    """
    if model is None or scaler is None:
        return {"status": "unhealthy", "message": "Mô hình hoặc bộ chuẩn hóa chưa được tải thành công."}
    return {"status": "healthy"}


@router.get("/model-info")
def get_model_info():
    """
    Endpoint 2: Trả về thông tin metadata và siêu tham số mô hình hiện tại
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng.")
    return {
        "Algorithm": "K-Means",
        "Số cụm": int(model.n_clusters),
        "Chỉ số Silhouette": 0.3200,  
        "Davies-Bouldin Index": 1.0172
    }


@router.post("/predict")
def predict_customer_segment(data: SinglePredictRequest):
    """
    Endpoint 3: Nhận thông tin thô, tiền xử lý, lưu DB và dự đoán nhóm cụm.
 
    kèm theo đánh giá độ tin cậy (confidence) dựa trên khoảng cách từ điểm dữ liệu
    tới tâm cụm được gán, so với ngưỡng học được lúc train. Khách hàng có hành vi quá khác
    biệt (outlier chưa từng thấy lúc train) sẽ được cảnh báo rõ thay vì trả lời chắc nịch sai.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng, thiếu file mô hình.")
 
    try:
        # Bước 3.1: Chuyển đổi chuỗi ngày nhận từ người dùng thành đối tượng datetime
        try:
            user_date = datetime.strptime(data.last_purchase_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Định dạng ngày không hợp lệ. Vui lòng dùng YYYY-MM-DD.")
 
        # Bước 3.2: Tính Recency 
        recency_days = (NOW - user_date).days
        if recency_days < 0:
            raise HTTPException(status_code=400, detail="Ngày mua hàng cuối cùng không thể ở tương lai.")
 
        # Bước 3.3: Tiền xử lý dữ liệu: log1p -> scale Z-score
        raw_features = pd.DataFrame(
            [[float(recency_days), data.frequency, data.monetary]],
            columns=['Recency', 'Frequency', 'Monetary']
        )
        log_features = raw_features.apply(np.log1p)
        scaled_features = scaler.transform(log_features)
 
        # Bước 3.4: Đưa vào mô hình K-Means dự đoán Cluster ID
        cluster_id = int(model.predict(scaled_features)[0])
 
        # MỚI - Bước 3.4b: Đánh giá độ tin cậy dựa trên khoảng cách tới tâm cụm được gán
        distances_to_all_centers = model.transform(scaled_features)[0]
        distance_to_assigned_center = float(distances_to_all_centers[cluster_id])
 
        threshold = cluster_thresholds.get(str(cluster_id))
        if threshold is not None:
            is_outlier = distance_to_assigned_center > threshold
            confidence = "low" if is_outlier else "high"
        else:
            is_outlier = None
            confidence = "unknown"
 
        # Bước 3.5: Ánh xạ Cluster ID sang định danh Persona kinh doanh
        key = str(cluster_id)
        if key in persona_map:
            persona_info = persona_map[key]
        else:
            persona_info = {"persona": f"Nhóm khách hàng {cluster_id}", "desc": "N/A"}
 
        persona_name = persona_info["persona"]
        description = persona_info.get("description", persona_info.get("desc", ""))
 
        # MỚI: nếu độ tin cậy thấp, thêm cảnh báo rõ ràng vào cả persona lẫn description
        if is_outlier:
            persona_name = f"{persona_name} (Ngoại lệ)"
            description = (
                "Hành vi mua hàng của khách này khác biệt đáng kể so với mọi khách hàng "
                "trong dữ liệu huấn luyện (Recency/Frequency/Monetary nằm ngoài phạm vi "
                "thông thường của cụm gần nhất). Kết quả phân cụm chỉ mang tính tham khảo, "
                "nên xem xét thủ công thay vì áp dụng chiến lược marketing hàng loạt. "
                + description
            )
 
        # Bước 3.6: Gọi hàm lưu lịch sử dự đoán vào PostgreSQL
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
            "distance_to_cluster_center": round(distance_to_assigned_center, 4),
            "distance_threshold": threshold
        }
 
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống trong quá trình xử lý: {str(e)}")


@router.get("/history")
def get_prediction_history():
    try:
        history_rows = get_prediction_history_from_db()
        return {"history": history_rows}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi truy xuất lịch sử: {str(e)}"
        )