import os
import joblib
import numpy as np
import pandas as pd
import io
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import json as _json

# ---- IMPORT TRỰC TIẾP CÁC HÀM TỪ FILE tiền XL ----
from src.preprocess import clean_data, build_rfm_features

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

cluster_thresholds = {}
if os.path.exists(THRESHOLD_JSON_PATH):
    try:
        with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
            cluster_thresholds = _json.load(f)
        print("[Backend] Đã nạp ngưỡng khoảng cách theo cụm thành công.")
    except Exception as e:
        print(f"[Lỗi] Không thể đọc file JSON ngưỡng khoảng cách: {e}")


# -------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA ĐỊNH DẠNG DỮ LIỆU ĐẦU VÀO CHO DỰ ĐOÁN ĐƠN LẺ
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

        raw_features = pd.DataFrame(
            [[float(recency_days), data.frequency, data.monetary]],
            columns=['Recency', 'Frequency', 'Monetary']
        )
        log_features = raw_features.apply(np.log1p)
        scaled_features = scaler.transform(log_features)

        cluster_id = int(model.predict(scaled_features)[0])

        distances_to_all_centers = model.transform(scaled_features)[0]
        distance_to_assigned_center = float(distances_to_all_centers[cluster_id])
        threshold = cluster_thresholds.get(str(cluster_id))
        confidence = "high" if threshold and distance_to_assigned_center <= threshold else "low"

        persona_info = persona_map.get(str(cluster_id), {"persona": f"Nhóm khách hàng {cluster_id}", "description": "N/A"})

        log_to_db(float(recency_days), data.frequency, data.monetary, cluster_id, persona_info["persona"])

        return {
            "cluster_id": cluster_id,
            "persona": persona_info["persona"],
            "description": persona_info.get("description", ""),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-file")
def predict_batch_csv(file: UploadFile = File(...)):
    """
    Endpoint nhận file CSV thô, TÁI SỬ DỤNG trực tiếp các hàm logic từ file tiền xử lý
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng, thiếu file mô hình.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Hệ thống chỉ hỗ trợ xử lý file định dạng .csv")

    try:
        # 1. Đọc file CSV thô từ bộ nhớ thành DataFrame
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Đồng bộ/Chuẩn hóa tên cột linh hoạt của người dùng về chữ thường
        df.columns = [col.strip().lower() for col in df.columns]

        mapping_dict = {
            'invoiceno': ['invoiceno', 'invoice_no', 'ma_hoa_don', 'mahoadon', 'id_hoa_don'],
            'quantity': ['quantity', 'so_luong', 'soluong', 'qty'],
            'invoicedate': ['invoicedate', 'invoice_date', 'ngay_hoa_don', 'ngayhoadon', 'ngay_mua', 'date'],
            'unitprice': ['unitprice', 'unit_price', 'don_gia', 'dongia', 'price'],
            'customerid': ['customerid', 'customer_id', 'ma_khach_hang', 'makhachhang', 'id_khach_hang']
        }

        # Ánh xạ tên cột linh hoạt về tên cột chuẩn mà hàm trong preprocess.py yêu cầu
        # (Lưu ý: preprocess.py viết hoa chữ cái đầu: InvoiceNo, Quantity, InvoiceDate, UnitPrice, CustomerID)
        standard_mapping = {
            'invoiceno': 'InvoiceNo',
            'quantity': 'Quantity',
            'invoicedate': 'InvoiceDate',
            'unitprice': 'UnitPrice',
            'customerid': 'CustomerID'
        }

        mapped_columns = {}
        for target_col, aliases in mapping_dict.items():
            found_col = next((col for col in df.columns if col in aliases), None)
            if found_col:
                mapped_columns[found_col] = standard_mapping[target_col]

        df.rename(columns=mapped_columns, inplace=True)

        # Kiểm tra tính hợp lệ của file thô
        if not all(col in df.columns for col in standard_mapping.values()):
            raise HTTPException(
                status_code=400, 
                detail="Cấu trúc file hóa đơn không tự động nhận diện được. Hãy chắc chắn chứa các cột cơ bản như Mã hóa đơn, Số lượng, Ngày, Đơn giá, Mã khách hàng."
            )

        # Chuyển đổi định dạng ngày tháng như hàm data_loader yêu cầu
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate'])

        # 3. GỌI TRỰC TIẾP HÀM LÀM SẠCH TỪ FILE PREPROCESS.PY
        cleaned_df = clean_data(df)

        # 4. GỌI TRỰC TIẾP HÀM GOM NHÓM RFM TỪ FILE PREPROCESS.PY
        # đọc tới chỗ này thì nhắn t giải thích
        # Ghi đè tạm thời hàm to_csv của hệ thống để nó không ghi gì ra ổ cứng nữa
        original_to_csv = pd.DataFrame.to_csv
        pd.DataFrame.to_csv = lambda *args, **kwargs: print("-> API chặn không ghi file rác ra ổ cứng.")
        
        try:
            # Chạy hàm công thức cũ, lúc này lệnh lưu file bên trong sẽ bị vô hiệu hóa
            rfm = build_rfm_features(cleaned_df, "data/dummy_path", snapshot_date=NOW)
        finally:
            # Trả lại hàm to_csv nguyên bản cho hệ thống sau khi dùng xong
            pd.DataFrame.to_csv = original_to_csv

        if rfm.empty:
            raise HTTPException(status_code=400, detail="Không có dữ liệu khách hàng hợp lệ sau khi làm sạch.")

        # 5. BIẾN ĐỔI LOG VÀ SCALE CHUẨN HÓA DỰA TRÊN SỰ CÓ SẴN CỦA MẢNG MỚI
        X_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
        X_scaled = scaler.transform(X_log)

        # 6. Đưa vào mô hình dự đoán
        rfm['Cluster'] = model.predict(X_scaled)
        distances_to_centers = model.transform(X_scaled)

        # 7. Đóng gói kết quả đầu ra
        batch_results = []
        for idx, row in rfm.iterrows():
            cluster_id = int(row['Cluster'])
            cust_id = str(row['CustomerID'])
            
            dist = float(distances_to_centers[len(batch_results), cluster_id])
            threshold = cluster_thresholds.get(str(cluster_id))
            confidence = "high" if threshold and dist <= threshold else "low"

            persona_info = persona_map.get(str(cluster_id), {"persona": f"Nhóm khách hàng {cluster_id}", "description": "N/A"})

            log_to_db(
                recency=float(row['Recency']),
                frequency=float(row['Frequency']),
                monetary=float(row['Monetary']),
                cluster_id=cluster_id,
                cluster_label=persona_info["persona"]
            )

            batch_results.append({
                "customer_id": cust_id,
                "recency_days": int(row['Recency']),
                "frequency_orders": int(row['Frequency']),
                "monetary_value": float(row['Monetary']),
                "cluster_id": cluster_id,
                "persona": persona_info["persona"],
                "description": persona_info.get("description", ""),
                "confidence": confidence
            })

        return {"total_customers_segmented": len(batch_results), "results": batch_results}

    except Exception as e:
        import traceback

        traceback.print_exc()

        if isinstance(e, HTTPException):
            raise e

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống khi xử lý tệp: {str(e)}"
    )


@router.get("/history")
def get_prediction_history():
    try:
        history_rows = get_prediction_history_from_db()
        return {"history": history_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi truy xuất lịch sử: {str(e)}")