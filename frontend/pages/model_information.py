from pathlib import Path

import streamlit as st

from utils.api_client import get_model_information


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ELBOW_IMAGE_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "elbow.png"
)

PCA_IMAGE_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "kmeans_pca_scatter_k4.png"
)

SILHOUETTE_IMAGE_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "silhouette_analysis_k4.png"
)


# =========================================================
# 2. THÔNG TIN MODEL CỐ ĐỊNH
# =========================================================
MODEL_VERSION = "2.0"
MODEL_ALGORITHM = "K-Means"
MODEL_UPDATED_DATE = "17/07/2026"

MODEL_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
]


# =========================================================
# 3. HÀM HỖ TRỢ
# =========================================================
def format_metric(value):
    """
    Hiển thị chỉ số dưới dạng 4 chữ số thập phân.
    """

    if value is None:
        return "Chưa có dữ liệu"

    try:
        return f"{float(value):.4f}"

    except (TypeError, ValueError):
        return str(value)



def get_api_value(
    data,
    possible_keys,
    default=None,
):
    """
    Lấy dữ liệu từ API và hỗ trợ nhiều cách đặt tên key.

    Ví dụ:
    silhouette_score
    Chỉ số Silhouette
    """

    for key in possible_keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def show_model_image(
    image_path,
    missing_message,
):
    """
    Hiển thị ảnh nếu file tồn tại.
    """

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True,
        )

    else:
        st.warning(
            missing_message
        )

        st.caption(
            f"Đường dẫn đang kiểm tra: `{image_path}`"
        )


# =========================================================
# 4. TIÊU ĐỀ
# =========================================================
st.title("Thông tin mô hình")
    

# =========================================================
# 5. GỌI API
# =========================================================
with st.spinner(
    "Đang tải thông tin mô hình từ backend..."
):
    api_result = get_model_information()


if not api_result.get("success"):
    st.error(
        api_result.get(
            "error",
            "Không thể lấy thông tin mô hình.",
        )
    )


    st.stop()


model_info = api_result.get(
    "data",
    {},
)


# =========================================================
# 6. LẤY DỮ LIỆU TỪ API
# =========================================================
number_of_clusters = get_api_value(
    model_info,
    [
        "n_clusters",
        "Số cụm",
        "number_of_clusters",
    ],
    default=4,
)

silhouette_score = get_api_value(
    model_info,
    [
        "silhouette_score",
        "Chỉ số Silhouette",
        "Silhouette Score",
    ],
)

davies_bouldin_index = get_api_value(
    model_info,
    [
        "davies_bouldin_index",
        "Davies-Bouldin Index",
        "davies_bouldin_score",
    ],
)


# =========================================================
# 7. TRẠNG THÁI TỔNG QUAN
# =========================================================
st.subheader("Trạng thái mô hình")

st.metric(
    label="Trạng thái",
    value="🟢 Đang hoạt động",
)

st.divider()


# =========================================================
# 8. CẤU HÌNH MÔ HÌNH
# =========================================================
st.subheader("Cấu hình mô hình")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model version", MODEL_VERSION)
col2.metric("Thuật toán", MODEL_ALGORITHM)
col3.metric("Số cụm", number_of_clusters)
col4.metric("Ngày cập nhật", MODEL_UPDATED_DATE)



st.divider()


# =========================================================
# 9. CHỈ SỐ ĐÁNH GIÁ
# =========================================================
st.subheader("Chỉ số đánh giá mô hình")

metric_col_1, metric_col_2 = st.columns(
    2,
    gap="large",
)


with metric_col_1:
    with st.container(border=True):
        st.markdown(
            "### Silhouette Score"
        )

        st.metric(
            label="Giá trị hiện tại",
            value=format_metric(
                silhouette_score
            ),
        )



with metric_col_2:
    with st.container(border=True):
        st.markdown(
            "### Davies-Bouldin Index"
        )

        st.metric(
            label="Giá trị hiện tại",
            value=format_metric(
                davies_bouldin_index
            ),
        )



st.divider()


# =========================================================
# 10. ELBOW METHOD
# =========================================================
st.subheader("Phân tích lựa chọn số cụm")

with st.container(border=True):
    st.markdown("### Elbow Method")

    st.caption(
        "Biểu đồ thể hiện sự thay đổi của Inertia "
        "khi số cụm K tăng lên. Điểm khuỷu được sử dụng "
        "để hỗ trợ lựa chọn số cụm phù hợp."
    )

    show_model_image(
        ELBOW_IMAGE_PATH,
        "Chưa tìm thấy file `notebooks/elbow.png`.",
    )


st.divider()


# =========================================================
# 11. PCA VÀ SILHOUETTE
# =========================================================
st.subheader("Trực quan hóa kết quả phân cụm")

chart_col_1, chart_col_2 = st.columns(
    2,
    gap="large",
)


with chart_col_1:
    with st.container(border=True):
        st.markdown(
            "### Phân bố cụm bằng PCA"
        )

        st.caption(
            "Biểu đồ trực quan hóa vị trí của các nhóm "
            "khách hàng sau khi dữ liệu được giảm xuống "
            "hai chiều bằng PCA."
        )

        show_model_image(
            PCA_IMAGE_PATH,
            (
                "Chưa tìm thấy file "
                "`notebooks/kmeans_pca_scatter_k4.png`."
            ),
        )


with chart_col_2:
    with st.container(border=True):
        st.markdown(
            "### Silhouette Analysis"
        )

        st.caption(
            "Biểu đồ thể hiện mức độ phù hợp của từng "
            "điểm dữ liệu đối với cụm được gán."
        )

        show_model_image(
            SILHOUETTE_IMAGE_PATH,
            (
                "Chưa tìm thấy file "
                "`notebooks/silhouette_analysis_k4.png`."
            ),
        )


st.divider()


# =========================================================
# 12. GIẢI THÍCH RFM
# =========================================================
st.subheader("Các biến đầu vào RFM")

rfm_col_1, rfm_col_2, rfm_col_3 = (
    st.columns(3)
)


with rfm_col_1:
    with st.container(border=True):
        st.markdown("### Recency")

        st.write(
            "Số ngày kể từ lần mua hàng gần nhất "
            "của khách hàng."
        )

        st.caption(
            "Giá trị càng thấp thường thể hiện "
            "khách hàng mua hàng càng gần đây."
        )


with rfm_col_2:
    with st.container(border=True):
        st.markdown("### Frequency")

        st.write(
            "Tổng số lần khách hàng phát sinh "
            "giao dịch mua hàng."
        )

        st.caption(
            "Giá trị càng cao cho thấy khách hàng "
            "mua hàng càng thường xuyên."
        )


with rfm_col_3:
    with st.container(border=True):
        st.markdown("### Monetary")

        st.write(
            "Tổng giá trị chi tiêu của khách hàng "
            "trong toàn bộ dữ liệu giao dịch."
        )

        st.caption(
            "Giá trị càng cao thể hiện khách hàng "
            "đóng góp doanh thu càng lớn."
        )


st.divider()
