from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. THÔNG TIN MODEL DEMO
# =========================================================
MODEL_INFO = {
    "version": "v1.0.0-demo",
    "algorithm": "K-Means Clustering",
    "last_updated": date.today().strftime("%d/%m/%Y"),
    "n_clusters": 4,
    "features": "Recency, Frequency, Monetary",
    "training_samples": 4338,
    "status": "Demo Mode",
    "silhouette_score": 0.52,
    "davies_bouldin_index": 0.71,
}


# =========================================================
# 2. DỮ LIỆU DEMO CHO ELBOW CURVE
# =========================================================
ELBOW_DATA = pd.DataFrame(
    {
        "K": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Inertia": [
            1800,
            1210,
            820,
            665,
            575,
            520,
            488,
            468,
            451,
        ],
    }
)


# =========================================================
# 3. TẠO DỮ LIỆU DEMO CHO SILHOUETTE PLOT
# =========================================================
def create_demo_silhouette_data():
    rng = np.random.default_rng(42)

    return {
        0: np.sort(rng.uniform(0.36, 0.72, 35)),
        1: np.sort(rng.uniform(0.28, 0.66, 40)),
        2: np.sort(rng.uniform(0.20, 0.58, 32)),
        3: np.sort(rng.uniform(0.31, 0.70, 37)),
    }


# =========================================================
# 4. TIÊU ĐỀ
# =========================================================
st.title("Thông tin mô hình")

st.caption(
    "Theo dõi cấu hình, phiên bản và chất lượng của mô hình "
    "phân khúc khách hàng."
)

st.info(
    "Demo Mode: Các chỉ số và biểu đồ hiện là dữ liệu minh họa. "
    "Sau khi nhóm hoàn thành huấn luyện, cần thay bằng kết quả thật."
)


# =========================================================
# 5. TRẠNG THÁI MODEL
# =========================================================
status_col_1, status_col_2, status_col_3 = st.columns(3)

status_col_1.metric(
    "Trạng thái",
    MODEL_INFO["status"],
)

status_col_2.metric(
    "Phiên bản",
    MODEL_INFO["version"],
)

status_col_3.metric(
    "Số cụm",
    MODEL_INFO["n_clusters"],
)


# =========================================================
# 6. THÔNG TIN META
# =========================================================
st.subheader("Thông tin cấu hình")

meta_df = pd.DataFrame(
    {
        "Thuộc tính": [
            "Model Version",
            "Thuật toán",
            "Ngày cập nhật gần nhất",
            "Số cụm khách hàng",
            "Biến đầu vào",
            "Số mẫu huấn luyện",
        ],
        "Giá trị": [
            MODEL_INFO["version"],
            MODEL_INFO["algorithm"],
            MODEL_INFO["last_updated"],
            MODEL_INFO["n_clusters"],
            MODEL_INFO["features"],
            MODEL_INFO["training_samples"],
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
# 7. CHỈ SỐ ĐÁNH GIÁ
# =========================================================
st.subheader("Chỉ số đánh giá mô hình")

metric_col_1, metric_col_2 = st.columns(2)

metric_col_1.metric(
    "Silhouette Score",
    f"{MODEL_INFO['silhouette_score']:.2f}",
    help=(
        "Giá trị càng gần 1 thường cho thấy các cụm "
        "càng tách biệt và chặt chẽ."
    ),
)

metric_col_2.metric(
    "Davies-Bouldin Index",
    f"{MODEL_INFO['davies_bouldin_index']:.2f}",
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
        - Nhỏ hơn `0`: một số điểm có thể được gán sai cụm.

        **Davies-Bouldin Index**

        - Giá trị thấp thường tốt hơn.
        - Chỉ số đo mức độ tương đồng giữa các cụm.
        """
    )


# =========================================================
# 8. BIỂU ĐỒ
# =========================================================
st.subheader("Phân tích lựa chọn số cụm")

elbow_col, silhouette_col = st.columns(
    2,
    gap="large",
)


# =========================================================
# 9. ELBOW CURVE
# =========================================================
with elbow_col:
    with st.container(border=True):
        st.markdown("### Elbow Curve")

        st.caption(
            "Quan sát điểm gãy để lựa chọn số cụm K phù hợp."
        )

        fig_elbow, ax_elbow = plt.subplots(
            figsize=(7, 5)
        )

        ax_elbow.plot(
            ELBOW_DATA["K"],
            ELBOW_DATA["Inertia"],
            marker="o",
            linewidth=2,
        )

        ax_elbow.axvline(
            x=MODEL_INFO["n_clusters"],
            linestyle="--",
            label=(
                f"K được chọn = "
                f"{MODEL_INFO['n_clusters']}"
            ),
        )

        ax_elbow.set_xlabel("Số cụm K")
        ax_elbow.set_ylabel("Inertia")
        ax_elbow.set_title(
            "Elbow Method – dữ liệu demo"
        )
        ax_elbow.grid(alpha=0.25)
        ax_elbow.legend()

        st.pyplot(
            fig_elbow,
            use_container_width=True,
        )


# =========================================================
# 10. SILHOUETTE PLOT
# =========================================================
with silhouette_col:
    with st.container(border=True):
        st.markdown("### Silhouette Plot")

        st.caption(
            "Đánh giá mức độ gắn kết và tách biệt của từng cụm."
        )

        silhouette_data = (
            create_demo_silhouette_data()
        )

        fig_silhouette, ax_silhouette = plt.subplots(
            figsize=(7, 5)
        )

        y_lower = 10

        for cluster, values in silhouette_data.items():
            y_upper = y_lower + len(values)

            ax_silhouette.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                alpha=0.7,
            )

            ax_silhouette.text(
                -0.05,
                y_lower + len(values) / 2,
                str(cluster),
            )

            y_lower = y_upper + 10

        ax_silhouette.axvline(
            x=MODEL_INFO["silhouette_score"],
            linestyle="--",
            label="Điểm trung bình",
        )

        ax_silhouette.set_xlabel(
            "Silhouette Coefficient"
        )
        ax_silhouette.set_ylabel(
            "Cụm khách hàng"
        )
        ax_silhouette.set_title(
            "Silhouette Plot – dữ liệu demo"
        )
        ax_silhouette.legend()

        st.pyplot(
            fig_silhouette,
            use_container_width=True,
        )


# =========================================================
# 11. NHẬN XÉT MODEL
# =========================================================
st.subheader("Nhận xét mô hình")

with st.container(border=True):
    st.write(
        f"Mô hình hiện đang sử dụng thuật toán "
        f"**{MODEL_INFO['algorithm']}** với "
        f"**{MODEL_INFO['n_clusters']} cụm khách hàng**."
    )

    st.write(
        f"Silhouette Score demo đạt "
        f"**{MODEL_INFO['silhouette_score']:.2f}**, "
        "cho thấy mức độ phân tách giữa các cụm "
        "ở mức tương đối."
    )

    st.write(
        f"Davies-Bouldin Index demo là "
        f"**{MODEL_INFO['davies_bouldin_index']:.2f}**. "
        "Chỉ số này cần được thay bằng kết quả thật "
        "sau khi nhóm hoàn thành huấn luyện."
    )


# =========================================================
# 12. CHECKLIST KHI NỐI MODEL THẬT
# =========================================================
with st.expander(
    "Thông tin cần thay khi có model thật",
    expanded=False,
):
    st.markdown(
        """
        Khi nhóm hoàn thành mô hình, cần thay:

        - Model Version.
        - Ngày cập nhật gần nhất.
        - Số cụm K được chọn.
        - Số mẫu huấn luyện.
        - Silhouette Score thật.
        - Davies-Bouldin Index thật.
        - Dữ liệu `K` và `Inertia` của Elbow Curve.
        - Silhouette value của từng mẫu.
        """
    )