import os
import json as _json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from groq import Groq
from dotenv import load_dotenv


OPTIMAL_K = 4
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"


def load_processed_data(processed_dir):
    """Đọc dữ liệu đã chuẩn hóa (train) và RFM thô đã loại outlier (diễn giải cụm)."""
    X_df = pd.read_csv(os.path.join(processed_dir, "customer_segmentation_scaled.csv"))
    X = X_df[['Recency', 'Frequency', 'Monetary']].values
    rfm_raw = pd.read_csv(os.path.join(processed_dir, "customer_segmentation.csv"))
    print(f" -> Kích thước ma trận huấn luyện: {X.shape} | Số khách hàng (đã loại outlier): {rfm_raw.shape[0]:,}")
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


def predict_full_customers(processed_dir, models_dir, model, rfm_raw):
    """Dự đoán cụm cho TOÀN BỘ khách hàng, kể cả outlier bị loại lúc train."""
    print("\n=== DỰ ĐOÁN CỤM CHO TOÀN BỘ KHÁCH HÀNG (BAO GỒM OUTLIER) ===")
    rfm_full = pd.read_csv(os.path.join(processed_dir, "customer_segmentation_full.csv"))
    scaler = joblib.load(os.path.join(models_dir, "rfm_scaler.pkl"))

    rfm_full_log = rfm_full[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    X_full_scaled = scaler.transform(rfm_full_log)
    rfm_full['Cluster'] = model.predict(X_full_scaled)
    rfm_full['Is_Outlier'] = ~rfm_full['CustomerID'].isin(rfm_raw['CustomerID'])

    print(f" -> Đã dự đoán cho {rfm_full.shape[0]:,} khách hàng "
          f"({rfm_full['Is_Outlier'].sum():,} outlier lúc train).")
    return rfm_full


def assign_persona_with_llm(rfm_full, models_dir):
    """Gọi Groq API MỘT LẦN để đặt tên persona, lưu mapping ra JSON để inference.py dùng lại
    (API production sẽ đọc file JSON này, KHÔNG gọi LLM lại mỗi request)."""
    print("\n=== GÁN TÊN PERSONA (GROQ API — CHỈ GỌI 1 LẦN) ===")
    load_dotenv()

    cluster_stats = rfm_full[rfm_full['Cluster'] != -1].groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_stats['SoLuongKhach'] = rfm_full[rfm_full['Cluster'] != -1].groupby('Cluster').size()
    cluster_stats['rank_recency'] = cluster_stats['Recency'].rank(ascending=True)
    cluster_stats['rank_frequency'] = cluster_stats['Frequency'].rank(ascending=False)
    cluster_stats['rank_monetary'] = cluster_stats['Monetary'].rank(ascending=False)
    cluster_stats['rank_score'] = cluster_stats[['rank_recency', 'rank_frequency', 'rank_monetary']].mean(axis=1)
    cluster_stats = cluster_stats.sort_values('rank_score')
    print(cluster_stats[['Recency', 'Frequency', 'Monetary', 'SoLuongKhach']])

    fallback_persona_pool = [
        "Khách hàng VIP (Champions)", "Khách hàng trung thành", "Khách hàng tiềm năng",
        "Khách hàng cần chăm sóc", "Khách hàng có nguy cơ rời bỏ", "Khách hàng không hoạt động"
    ]

    cluster_description = "\n".join([
        f"- Cluster {cid}: Recency trung bình = {row['Recency']:.1f} ngày, "
        f"Frequency trung bình = {row['Frequency']:.1f} đơn hàng, "
        f"Monetary trung bình = {row['Monetary']:,.0f}, số lượng khách hàng = {int(row['SoLuongKhach'])}"
        for cid, row in cluster_stats.iterrows()
    ])

    prompt = f"""Bạn là chuyên gia phân tích khách hàng (CRM/Marketing). Dưới đây là kết quả phân cụm khách hàng theo mô hình RFM (Recency = số ngày từ lần mua gần nhất, càng THẤP càng tốt; Frequency = số đơn hàng, càng CAO càng tốt; Monetary = tổng chi tiêu, càng CAO càng tốt):

{cluster_description}

Với MỖI cluster ở trên, hãy đặt:
1. "persona": một tên phân khúc khách hàng ngắn gọn bằng tiếng Việt (3-6 từ)
2. "description": mô tả ngắn 1 câu về đặc điểm hành vi và gợi ý chiến lược chăm sóc/marketing phù hợp

Chỉ trả về JSON hợp lệ theo đúng định dạng sau, không thêm giải thích hay markdown:
{{"0": {{"persona": "...", "description": "..."}}, "1": {{"persona": "...", "description": "..."}}}}
"""

    cluster_to_persona = {}
    cluster_to_description = {}

    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Không tìm thấy GROQ_API_KEY")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        llm_result = _json.loads(response.choices[0].message.content)

        for cid in cluster_stats.index:
            key = str(cid)
            if key in llm_result and "persona" in llm_result[key]:
                cluster_to_persona[cid] = llm_result[key]["persona"]
                cluster_to_description[cid] = llm_result[key].get("description", "")
            else:
                raise KeyError(f"Thiếu kết quả cho cluster {cid}")
        print(" -> [Thành công] Đã gọi Groq API để định danh persona.")

    except Exception as e:
        print(f" -> [Thất bại] Groq API lỗi ({e}) -> dùng nhãn dự phòng.")
        cluster_to_persona = {
            cid: (fallback_persona_pool[i] if i < len(fallback_persona_pool) else f"Nhóm khách hàng {i + 1}")
            for i, cid in enumerate(cluster_stats.index)
        }
        cluster_to_description = {cid: "Chưa có mô tả do lỗi kết nối AI." for cid in cluster_stats.index}

    # Lưu mapping ra JSON để inference.py dùng lại, không cần gọi LLM mỗi request
    persona_mapping = {
        str(cid): {"persona": cluster_to_persona[cid], "description": cluster_to_description[cid]}
        for cid in cluster_to_persona
    }
    persona_path = os.path.join(models_dir, "persona_mapping.json")
    with open(persona_path, "w", encoding="utf-8") as f:
        _json.dump(persona_mapping, f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu persona mapping tại: {persona_path}")

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
    rfm_full = predict_full_customers(PROCESSED_DATA_DIR, MODELS_DIR, model, rfm_raw)
    rfm_full = assign_persona_with_llm(rfm_full, MODELS_DIR)
    export_final_results(rfm_full, PROCESSED_DATA_DIR)

