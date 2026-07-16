from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


# =========================================================
# 1. ĐƯỜNG DẪN
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "rfm_scaler.pkl"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segmentation_full.csv"
)

RFM_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
]


# =========================================================
# 2. TẢI MODEL VÀ DỮ LIỆU THẬT
# =========================================================
@st.cache_resource
def load_model_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy scaler tại: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


@st.cache_data
def load_rfm_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
    )

    missing_columns = [
        column
        for column in RFM_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dữ liệu đang thiếu các cột: "
            + ", ".join(missing_columns)
        )

    rfm_df = df[RFM_COLUMNS].copy()

    for column in RFM_COLUMNS:
        rfm_df[column] = pd.to_numeric(
            rfm_df[column],
            errors="coerce",
        )

    rfm_df = rfm_df.dropna()

    # Đảm bảo dữ liệu hợp lệ trước log1p
    rfm_df = rfm_df[
        (rfm_df["Recency"] >= 0)
        & (rfm_df["Frequency"] >= 0)
        & (rfm_df["Monetary"] >= 0)
    ]

    return rfm_df


try:
    model, scaler = load_model_assets()
    rfm_df = load_rfm_data()

except Exception as error:
    st.error(f"Không thể tải thông tin mô hình: {error}")
    st.stop()


# =========================================================
# 3. ÁP DỤNG ĐÚNG PIPELINE ĐÃ HUẤN LUYỆN
# =========================================================
rfm_log = np.log1p(rfm_df[RFM_COLUMNS])

rfm_scaled = scaler.transform(rfm_log)

cluster_labels = model.predict(rfm_scaled)

n_clusters = int(model.n_clusters)

training_samples = len(rfm_df)

silhouette_value = silhouette_score(
    rfm_scaled,
    cluster_labels,
)

davies_bouldin_value = davies_bouldin_score(
    rfm_scaled,
    cluster_labels,
)

model_updated_time = datetime.fromtimestamp(
    MODEL_PATH.stat().st_mtime
).strftime("%d/%m/%Y %H:%M")


# =========================================================
# 4. THÔNG TIN MODEL THẬT
# =========================================================
MODEL_INFO = {
    "version": MODEL_PATH.stem,
    "algorithm": type(model).__name__,
    "last_updated": model_updated_time,
    "n_clusters": n_clusters,
    "features": ", ".join(RFM_COLUMNS),
    "training_samples": training_samples,
    "status": "Đã kết nối",
    "silhouette_score": silhouette_value,
    "davies_bouldin_index": davies_bouldin_value,
}


# =========================================================
# 5. TIÊU ĐỀ
# =========================================================
st.title("Thông tin mô hình")

st.caption(
    "Theo dõi cấu hình và chất lượng của mô hình "
    "phân khúc khách hàng đang được hệ thống sử dụng."
)


# =========================================================
# 6. TRẠNG THÁI MODEL
# =========================================================
status_col_1, status_col_2, status_col_3 = st.columns(3)

status_col_1.metric(
    "Trạng thái",
    MODEL_INFO["status"],
)

status_col_2.metric(
    "Mô hình",
    MODEL_INFO["algorithm"],
)

status_col_3.metric(
    "Số cụm",
    MODEL_INFO["n_clusters"],
)


# =========================================================
# 7. THÔNG TIN CẤU HÌNH
# =========================================================
st.subheader("Thông tin cấu hình")

meta_df = pd.DataFrame(
    {
        "Thuộc tính": [
            "Tên file model",
            "Thuật toán",
            "Ngày cập nhật gần nhất",
            "Số cụm khách hàng",
            "Biến đầu vào",
            "Số mẫu đánh giá",
            "File dữ liệu",
        ],
        "Giá trị": [
            str(MODEL_INFO["version"]),
            str(MODEL_INFO["algorithm"]),
            str(MODEL_INFO["last_updated"]),
            str(MODEL_INFO["n_clusters"]),
            str(MODEL_INFO["features"]),
            str(MODEL_INFO["training_samples"]),
            str(DATA_PATH.name),
        ],
    }
)

st.dataframe(
    meta_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Thuộc tính": st.column_config.TextColumn(
            "Thuộc tính",
            width="medium",
        ),
        "Giá trị": st.column_config.TextColumn(
            "Giá trị",
            width="large",
        ),
    },
)


# =========================================================
# 8. CHỈ SỐ ĐÁNH GIÁ THẬT
# =========================================================
st.subheader("Chỉ số đánh giá mô hình")

metric_col_1, metric_col_2 = st.columns(2)

metric_col_1.metric(
    "Silhouette Score",
    f"{silhouette_value:.4f}",
    help=(
        "Giá trị càng gần 1 cho thấy các cụm "
        "càng chặt chẽ và tách biệt."
    ),
)

metric_col_2.metric(
    "Davies-Bouldin Index",
    f"{davies_bouldin_value:.4f}",
    help=(
        "Giá trị càng thấp thường cho thấy "
        "chất lượng phân cụm càng tốt."
    ),
)

with st.expander(
    "Xem cách diễn giải các chỉ số",
    expanded=False,
):
    st.markdown(
        """
        **Silhouette Score**

        - Gần `1`: các cụm tách biệt tốt.
        - Gần `0`: các cụm có thể chồng lấn.
        - Nhỏ hơn `0`: một số điểm có thể được gán chưa phù hợp.

        **Davies-Bouldin Index**

        - Giá trị càng thấp thường càng tốt.
        - Chỉ số phản ánh mức độ tương đồng giữa các cụm.
        """
    )


# =========================================================
# 9. TÍNH ELBOW CURVE TỪ DỮ LIỆU THẬT
# =========================================================
@st.cache_data
def calculate_elbow_data(
    scaled_data: np.ndarray,
) -> pd.DataFrame:
    max_k = min(10, len(scaled_data) - 1)

    k_values = list(range(2, max_k + 1))
    inertia_values = []

    for k in k_values:
        candidate_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        candidate_model.fit(scaled_data)

        inertia_values.append(
            float(candidate_model.inertia_)
        )

    return pd.DataFrame(
        {
            "K": k_values,
            "Inertia": inertia_values,
        }
    )


elbow_data = calculate_elbow_data(rfm_scaled)


# =========================================================
# 10. BIỂU ĐỒ
# =========================================================
st.subheader("Phân tích lựa chọn số cụm")

elbow_col, silhouette_col = st.columns(
    2,
    gap="large",
)


# =========================================================
# 11. ELBOW CURVE THẬT
# =========================================================
with elbow_col:
    with st.container(border=True):
        st.markdown("### Elbow Curve")

        st.caption(
            "Được tính trực tiếp từ dữ liệu RFM đã tiền xử lý."
        )

        fig_elbow, ax_elbow = plt.subplots(
            figsize=(7, 5)
        )

        ax_elbow.plot(
            elbow_data["K"],
            elbow_data["Inertia"],
            marker="o",
            linewidth=2,
        )

        ax_elbow.axvline(
            x=n_clusters,
            linestyle="--",
            label=f"K đang sử dụng = {n_clusters}",
        )

        ax_elbow.set_xlabel("Số cụm K")
        ax_elbow.set_ylabel("Inertia")
        ax_elbow.set_title(
            "Elbow Method – dữ liệu thực tế"
        )
        ax_elbow.grid(alpha=0.25)
        ax_elbow.legend()

        st.pyplot(
            fig_elbow,
            use_container_width=True,
        )

        plt.close(fig_elbow)


# =========================================================
# 12. SILHOUETTE PLOT THẬT
# =========================================================
with silhouette_col:
    with st.container(border=True):
        st.markdown("### Silhouette Plot")

        st.caption(
            "Thể hiện mức độ gắn kết và tách biệt "
            "của từng cụm do model dự đoán."
        )

        silhouette_values = silhouette_samples(
            rfm_scaled,
            cluster_labels,
        )

        fig_silhouette, ax_silhouette = plt.subplots(
            figsize=(7, 5)
        )

        y_lower = 10

        for cluster in range(n_clusters):
            cluster_silhouette_values = (
                silhouette_values[
                    cluster_labels == cluster
                ]
            )

            cluster_silhouette_values.sort()

            cluster_size = len(
                cluster_silhouette_values
            )

            y_upper = y_lower + cluster_size

            ax_silhouette.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                alpha=0.7,
            )

            ax_silhouette.text(
                -0.05,
                y_lower + cluster_size / 2,
                str(cluster),
            )

            y_lower = y_upper + 10

        ax_silhouette.axvline(
            x=silhouette_value,
            linestyle="--",
            label=(
                f"Điểm trung bình: "
                f"{silhouette_value:.4f}"
            ),
        )

        ax_silhouette.set_xlabel(
            "Silhouette Coefficient"
        )

        ax_silhouette.set_ylabel(
            "Cụm khách hàng"
        )

        ax_silhouette.set_title(
            "Silhouette Plot – dữ liệu thực tế"
        )

        ax_silhouette.legend()

        st.pyplot(
            fig_silhouette,
            use_container_width=True,
        )

        plt.close(fig_silhouette)


# =========================================================
# 13. NHẬN XÉT MODEL
# =========================================================
st.subheader("Nhận xét mô hình")

with st.container(border=True):
    st.write(
        f"Hệ thống đang sử dụng mô hình "
        f"**{MODEL_INFO['algorithm']}** với "
        f"**{n_clusters} cụm khách hàng**."
    )

    st.write(
        f"Silhouette Score hiện tại đạt "
        f"**{silhouette_value:.4f}**."
    )

    st.write(
        f"Davies-Bouldin Index hiện tại là "
        f"**{davies_bouldin_value:.4f}**."
    )

    st.write(
        f"Các chỉ số trên được tính từ "
        f"**{training_samples:,} bản ghi RFM** "
        "sau khi áp dụng cùng pipeline tiền xử lý "
        "với mô hình."
    )


# =========================================================
# 14. XÁC NHẬN FILE ĐANG SỬ DỤNG
# =========================================================
with st.expander(
    "Kiểm tra nguồn mô hình",
    expanded=False,
):
    st.write("**File model:**", str(MODEL_PATH))
    st.write("**File scaler:**", str(SCALER_PATH))
    st.write("**File dữ liệu:**", str(DATA_PATH))
    st.write("**Loại model:**", type(model).__name__)
    st.write("**Loại scaler:**", type(scaler).__name__)