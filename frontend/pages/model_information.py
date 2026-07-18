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


def evaluate_silhouette(value):
    """
    Nhận xét Silhouette Score.
    """

    if value is None:
        return (
            "Chưa nhận được Silhouette Score từ API."
        )

    try:
        value = float(value)

    except (TypeError, ValueError):
        return (
            "Silhouette Score trả về không đúng định dạng."
        )

    if value >= 0.50:
        return (
            "Các cụm có mức độ gắn kết và phân tách tốt."
        )

    if value >= 0.25:
        return (
            "Mô hình có cấu trúc phân cụm ở mức chấp nhận được, "
            "tuy nhiên một số cụm vẫn có thể chồng lấn."
        )

    return (
        "Các cụm có khả năng chồng lấn nhiều. "
        "Nên kiểm tra lại đặc trưng đầu vào hoặc số cụm."
    )


def evaluate_davies_bouldin(value):
    """
    Nhận xét Davies-Bouldin Index.
    """

    if value is None:
        return (
            "Chưa nhận được Davies-Bouldin Index từ API."
        )

    try:
        value = float(value)

    except (TypeError, ValueError):
        return (
            "Davies-Bouldin Index trả về không đúng định dạng."
        )

    if value < 0.50:
        return (
            "Các cụm có mức độ phân tách tốt."
        )

    if value < 1.00:
        return (
            "Chất lượng phân cụm tương đối tốt."
        )

    if value < 1.50:
        return (
            "Chất lượng phân cụm ở mức trung bình."
        )

    return (
        "Các cụm có thể chưa được phân tách rõ ràng."
    )


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

st.caption(
    "Theo dõi cấu hình, chất lượng và kết quả đánh giá "
    "của mô hình phân khúc khách hàng."
)


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

    st.info(
        "Hãy kiểm tra backend FastAPI đã được chạy chưa:\n\n"
        "`uvicorn backend.main:app --reload`"
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

status_col_1, status_col_2, status_col_3, status_col_4 = (
    st.columns(4)
)

status_col_1.metric(
    label="Trạng thái",
    value="Đã kết nối",
)

status_col_2.metric(
    label="Phiên bản",
    value=MODEL_VERSION,
)

status_col_3.metric(
    label="Thuật toán",
    value=MODEL_ALGORITHM,
)

status_col_4.metric(
    label="Số cụm",
    value=number_of_clusters,
)


st.divider()


# =========================================================
# 8. CẤU HÌNH MÔ HÌNH
# =========================================================
st.subheader("Cấu hình mô hình")

with st.container(border=True):
    config_col_1, config_col_2 = st.columns(
        2,
        gap="large",
    )

    with config_col_1:
        st.markdown("**Model version**")
        st.write(MODEL_VERSION)

        st.markdown("**Thuật toán**")
        st.write(MODEL_ALGORITHM)

        st.markdown("**Số cụm khách hàng**")
        st.write(number_of_clusters)

    with config_col_2:
        st.markdown("**Ngày cập nhật**")
        st.write(MODEL_UPDATED_DATE)

        st.markdown("**Biến đầu vào**")
        st.write(
            ", ".join(MODEL_FEATURES)
        )

        st.markdown("**Phương pháp tiền xử lý**")
        st.write(
            "Log Transformation và Standard Scaling"
        )


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

        st.caption(
            "Giá trị càng gần 1 cho thấy các điểm dữ liệu "
            "trong cùng cụm càng gần nhau và các cụm "
            "càng tách biệt."
        )

        st.write(
            evaluate_silhouette(
                silhouette_score
            )
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

        st.caption(
            "Giá trị càng thấp thường cho thấy "
            "chất lượng phân cụm càng tốt."
        )

        st.write(
            evaluate_davies_bouldin(
                davies_bouldin_index
            )
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


# =========================================================
# 13. NHẬN XÉT TỔNG QUAN
# =========================================================
st.subheader("Nhận xét tổng quan")

with st.container(border=True):
    st.write(
        f"Hệ thống hiện sử dụng mô hình "
        f"**{MODEL_ALGORITHM} phiên bản {MODEL_VERSION}** "
        f"để phân chia khách hàng thành "
        f"**{number_of_clusters} cụm**."
    )

    st.write(
        "Mô hình được xây dựng dựa trên ba đặc trưng "
        "**Recency, Frequency và Monetary**. "
        "Dữ liệu được biến đổi log và chuẩn hóa trước "
        "khi đưa vào mô hình K-Means."
    )

    if silhouette_score is not None:
        st.write(
            f"Silhouette Score hiện tại đạt "
            f"**{format_metric(silhouette_score)}**. "
            f"{evaluate_silhouette(silhouette_score)}"
        )

    if davies_bouldin_index is not None:
        st.write(
            f"Davies-Bouldin Index hiện tại là "
            f"**{format_metric(davies_bouldin_index)}**. "
            f"{evaluate_davies_bouldin(davies_bouldin_index)}"
        )


# =========================================================
# 14. KIỂM TRA NGUỒN DỮ LIỆU
# =========================================================
with st.expander(
    "Kiểm tra nguồn thông tin mô hình",
    expanded=False,
):
    st.write(
        "**API:** `/model-info`"
    )

    st.write(
        "**Ảnh Elbow:**",
        str(ELBOW_IMAGE_PATH),
    )

    st.write(
        "**Ảnh PCA:**",
        str(PCA_IMAGE_PATH),
    )

    st.write(
        "**Ảnh Silhouette:**",
        str(SILHOUETTE_IMAGE_PATH),
    )

    st.write(
        "**Dữ liệu API nhận được:**"
    )

    st.json(model_info)