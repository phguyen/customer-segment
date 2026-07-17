import os
import json as _json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

OPTIMAL_K = 4
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"


def load_processed_data(processed_dir):
    """Đọc dữ liệu đã chuẩn hóa (train) và RFM thô ."""
    X_df = pd.read_csv(os.path.join(processed_dir, "customer_segmentation_scaled.csv"))
    X = X_df[['Recency', 'Frequency', 'Monetary']].values
    rfm_raw = pd.read_csv(os.path.join(processed_dir, "customer_segmentation.csv"))
    print(f" -> Kích thước ma trận huấn luyện: {X.shape} | Số khách hàng: {rfm_raw.shape[0]:,}")
    return X, rfm_raw


def train_kmeans(X, k, models_dir):
    """Huấn luyện K-Means với k đã chốt, lưu model."""
    print(f"\n=== HUẤN LUYỆN K-MEANS (k={k}) ===")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f" -> Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")

    model_path = os.path.join(models_dir, "kmeans_model.pkl")
    joblib.dump(model, model_path)
    print(f" -> Đã lưu model tại: {model_path}")

    return model, labels


def save_cluster_distance_thresholds(X, model, labels, models_dir, percentile=95):
    """Tính ngưỡng khoảng cách (tới tâm cụm) theo percentile cho từng cụm.
    Dùng ở inference để phát hiện khách hàng có hành vi bất thường/ngoại lệ."""
    print(f"\n=== TÍNH NGƯỠNG KHOẢNG CÁCH THEO CỤM (percentile={percentile}) ===")
    distances_to_all_centers = model.transform(X)  # shape (n_samples, n_clusters)

    thresholds = {}
    for cluster_id in range(model.n_clusters):
        mask = labels == cluster_id
        dist_to_own_center = distances_to_all_centers[mask, cluster_id]
        threshold = float(np.percentile(dist_to_own_center, percentile))
        thresholds[str(cluster_id)] = threshold
        print(f" -> Cụm {cluster_id}: ngưỡng khoảng cách (p{percentile}) = {threshold:.4f}")

    threshold_path = os.path.join(models_dir, "cluster_distance_thresholds.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        _json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu ngưỡng tại: {threshold_path}")

    return thresholds


def predict_full_customers(processed_dir, models_dir, model):
    """Dự đoán cụm cho TOÀN BỘ khách hàng."""
    print("\n=== DỰ ĐOÁN CỤM CHO TOÀN BỘ KHÁCH HÀNG ===")
    rfm_full = pd.read_csv(os.path.join(processed_dir, "customer_segmentation_full.csv"))
    scaler = joblib.load(os.path.join(models_dir, "rfm_scaler.pkl"))

    rfm_full_log = rfm_full[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    X_full_scaled = scaler.transform(rfm_full_log)
    rfm_full['Cluster'] = model.predict(X_full_scaled)

    print(f" -> Đã dự đoán cụm thành công cho {rfm_full.shape[0]:,} khách hàng.")
    return rfm_full


def assign_persona_from_json(rfm_full, models_dir):
    """Đọc cấu hình Persona mapping đã tồn tại từ file JSON có sẵn 
    thay vì gọi trực tiếp tới API Groq."""
    print("\n=== GÁN TÊN PERSONA TỪ FILE JSON CẤU HÌNH CÓ SẴN ===")
    
    persona_path = os.path.join(models_dir, "persona_mapping.json")
    
    if not os.path.exists(persona_path):
        raise FileNotFoundError(f"Không tìm thấy file JSON cấu hình nhân hóa tại: {persona_path}. Vui lòng tạo file trước.")
        
    with open(persona_path, "r", encoding="utf-8") as f:
        persona_mapping = _json.load(f)
    print(f" -> Đã nạp thành công file cấu hình: {persona_path}")

    # Khởi tạo các bộ từ điển để map thông tin dựa trên file JSON có sẵn
    cluster_to_persona = {}
    cluster_to_description = {}

    for key, value in persona_mapping.items():
        cid = int(key)
        cluster_to_persona[cid] = value.get("persona", f"Nhóm {cid}")
        cluster_to_description[cid] = value.get("description", "Không có mô tả.")

    rfm_full['Persona'] = rfm_full['Cluster'].map(cluster_to_persona)
    rfm_full['Persona_Description'] = rfm_full['Cluster'].map(cluster_to_description)
    return rfm_full




def export_final_results(rfm_full, processed_dir):
    final_output_path = os.path.join(processed_dir, "customer_segments_final.csv")
    rfm_full.to_csv(final_output_path, index=False, encoding="utf-8-sig")
    print(f"\n -> Đã lưu file kết quả cuối cùng tại: {final_output_path}")
    print(rfm_full['Persona'].value_counts())
    return final_output_path


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, rfm_raw = load_processed_data(PROCESSED_DATA_DIR)
    model, labels = train_kmeans(X, OPTIMAL_K, MODELS_DIR)
    save_cluster_distance_thresholds(X, model, labels, MODELS_DIR, percentile=95)
    
    # Dự đoán và mapping nhãn trực tiếp bằng tệp JSON có sẵn
    rfm_full = predict_full_customers(PROCESSED_DATA_DIR, MODELS_DIR, model)
    rfm_full = assign_persona_from_json(rfm_full, MODELS_DIR)
    
    export_final_results(rfm_full, PROCESSED_DATA_DIR)