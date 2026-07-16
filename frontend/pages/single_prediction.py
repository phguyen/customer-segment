from datetime import date, datetime
import time

import pandas as pd
import streamlit as st

from utils.history_utils import append_prediction_history
from utils.prediction_utils import predict_customer


# =========================================================
# 1. THÔNG TIN HIỂN THỊ THEO PHÂN KHÚC
# =========================================================
# CLUSTER_CONTENT = {
#     0: {
#         "status": "VIP",
#         "description": (
#             "Đây là nhóm khách hàng có giá trị cao, mua hàng gần đây, "
#             "tần suất mua tốt và đóng góp doanh thu lớn."
#         ),
#         "recommendations": [
#             "Ưu tiên chương trình chăm sóc khách hàng VIP",
#             "Cung cấp ưu đãi hoặc quyền lợi độc quyền",
#             "Cá nhân hóa nội dung và sản phẩm đề xuất",
#             "Khuyến khích giới thiệu khách hàng mới",
#         ],
#         "message_type": "success",
#     },
#     1: {
#         "status": "Potential",
#         "description": (
#             "Đây là nhóm khách hàng có mức độ tương tác tốt và còn nhiều "
#             "khả năng gia tăng tần suất mua hoặc tổng giá trị chi tiêu."
#         ),
#         "recommendations": [
#             "Khuyến khích mua lại bằng voucher",
#             "Đề xuất combo sản phẩm phù hợp",
#             "Gửi nội dung chăm sóc cá nhân hóa",
#             "Theo dõi phản hồi sau mua hàng",
#         ],
#         "message_type": "info",
#     },
#     2: {
#         "status": "Need Attention",
#         "description": (
#             "Đây là nhóm khách hàng có hành vi mua ở mức trung bình "
#             "và cần được tiếp tục nuôi dưỡng."
#         ),
#         "recommendations": [
#             "Gửi nội dung nhắc nhớ thương hiệu",
#             "Áp dụng ưu đãi nhẹ để kích thích mua lại",
#             "Gợi ý sản phẩm dựa trên lịch sử mua hàng",
#             "Theo dõi tần suất mua trong 30 ngày tới",
#         ],
#         "message_type": "warning",
#     },
#     3: {
#         "status": "At Risk",
#         "description": (
#             "Đây là nhóm khách hàng đã lâu chưa quay lại hoặc có tần suất "
#             "mua thấp, cần được ưu tiên tái kích hoạt."
#         ),
#         "recommendations": [
#             "Triển khai chiến dịch tái kích hoạt",
#             "Gửi ưu đãi quay lại có thời hạn",
#             "Khảo sát nguyên nhân khách hàng ngừng mua",
#             "Ưu tiên chăm sóc trong 7 ngày tới",
#         ],
#         "message_type": "error",
#     },
# }
CLUSTER_CONTENT = {
    0: {
        "description": (
            "Đây là nhóm khách hàng tiềm năng mới gia nhập, "
            "cần được khuyến khích mua thêm."
        ),
        "recommendations": [
            "Gửi ưu đãi cho lần mua tiếp theo",
            "Giới thiệu các sản phẩm phổ biến",
            "Hướng dẫn khách hàng làm quen với thương hiệu",
            "Theo dõi hành vi mua trong 30 ngày đầu",
        ],
        "message_type": "info",
    },

    1: {
        "description": (
            "Đây là nhóm khách hàng có nguy cơ rời bỏ, "
            "cần được ưu tiên tái kích hoạt."
        ),
        "recommendations": [
            "Triển khai chiến dịch tái kích hoạt",
            "Gửi ưu đãi quay lại có thời hạn",
            "Khảo sát nguyên nhân khách hàng ngừng mua",
            "Ưu tiên chăm sóc trong 7 ngày tới",
        ],
        "message_type": "error",
    },

    2: {
        "description": (
            "Đây là nhóm khách hàng có mức độ trung thành còn thấp "
            "và cần được tiếp tục chăm sóc để gia tăng giá trị."
        ),
        "recommendations": [
            "Gửi nội dung nhắc nhớ thương hiệu",
            "Áp dụng ưu đãi nhẹ để kích thích mua lại",
            "Gợi ý sản phẩm dựa trên lịch sử mua hàng",
            "Theo dõi tần suất mua trong 30 ngày tới",
        ],
        "message_type": "warning",
    },

    3: {
        "description": (
            "Đây là nhóm khách hàng giá trị cao nhất, "
            "cần được ưu tiên chăm sóc đặc biệt."
        ),
        "recommendations": [
            "Ưu tiên chương trình chăm sóc khách hàng VIP",
            "Cung cấp ưu đãi hoặc quyền lợi độc quyền",
            "Cá nhân hóa nội dung và sản phẩm đề xuất",
            "Khuyến khích giới thiệu khách hàng mới",
        ],
        "message_type": "success",
    },
}

def get_cluster_content(cluster: int) -> dict:
    """Lấy nội dung hiển thị tương ứng với từng cụm."""

    return CLUSTER_CONTENT.get(
        cluster,
        {
            "status": "Customer Segment",
            "description": (
                "Khách hàng đã được mô hình phân vào một nhóm hành vi "
                "dựa trên các chỉ số Recency, Frequency và Monetary."
            ),
            "recommendations": [
                "Theo dõi thêm hành vi mua hàng",
                "Cá nhân hóa nội dung chăm sóc",
                "Đề xuất sản phẩm phù hợp",
                "Đánh giá lại phân khúc định kỳ",
            ],
            "message_type": "info",
        },
    )


# =========================================================
# 2. CSS
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1280px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-weight: 760;
            letter-spacing: -0.5px;
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5EAF3;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(31, 42, 68, 0.05);
        }

        div[data-testid="stForm"] {
            border: none;
        }

        div.stFormSubmitButton > button {
            min-height: 48px;
            border-radius: 10px;
            font-weight: 700;
        }

        div[data-baseweb="input"] > div {
            border-radius: 10px;
        }

        div[data-baseweb="base-input"] > div {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. TIÊU ĐỀ
# =========================================================
st.title("Phân nhóm khách hàng đơn lẻ")

st.caption(
    "Xác định phân khúc khách hàng bằng mô hình K-Means "
    "dựa trên ba chỉ số Recency, Frequency và Monetary."
)


# =========================================================
# 4. BỐ CỤC
# =========================================================
input_col, result_col = st.columns(
    [1.02, 0.98],
    gap="large",
)


# =========================================================
# 5. FORM NHẬP DỮ LIỆU
# =========================================================
with input_col:
    with st.container(border=True):
        st.subheader("Thông tin khách hàng")

        st.caption(
            "Nhập thông tin mua hàng để hệ thống tính toán chỉ số RFM."
        )

        with st.form("single_prediction_form"):
            customer_id = st.text_input(
                "Mã khách hàng",
                placeholder="Ví dụ: CUST001",
                help="Mã định danh khách hàng trong hệ thống.",
            )

            last_purchase_date = st.date_input(
                "Ngày mua hàng gần nhất",
                value=date.today(),
                max_value=date.today(),
                format="DD/MM/YYYY",
                help="Ngày phát sinh giao dịch gần nhất của khách hàng.",
            )

            frequency = st.number_input(
                "Tổng số hóa đơn đã mua (Frequency)",
                min_value=1,
                value=1,
                step=1,
                help="Tổng số hóa đơn hoặc số lần mua của khách hàng.",
            )

            monetary = st.number_input(
                "Tổng số tiền đã chi tiêu (Monetary)",
                min_value=0.0,
                value=0.0,
                step=100_000.0,
                format="%.0f",
                help="Tổng giá trị chi tiêu của khách hàng.",
            )

            submitted = st.form_submit_button(
                "Phân tích khách hàng",
                type="primary",
                use_container_width=True,
            )


# =========================================================
# 6. KẾT QUẢ
# =========================================================
with result_col:
    with st.container(border=True):
        st.subheader("Kết quả phân tích")

        if not submitted:
            st.caption(
                "Kết quả sẽ xuất hiện sau khi bạn hoàn thành biểu mẫu."
            )

            st.info(
                "Chưa có dữ liệu phân tích. "
                "Hãy nhập thông tin khách hàng ở bên trái."
            )

            placeholder_1, placeholder_2 = st.columns(2)

            placeholder_1.metric("Mã cụm", "—")
            placeholder_2.metric("Recency", "—")

            st.divider()

            st.subheader("Đề xuất Marketing")

            st.write(
                "Đề xuất sẽ được hiển thị sau khi hệ thống "
                "xác định được phân khúc khách hàng."
            )

        else:
            errors = []

            if not customer_id.strip():
                errors.append("Vui lòng nhập mã khách hàng.")

            if monetary <= 0:
                errors.append(
                    "Tổng số tiền đã chi tiêu phải lớn hơn 0."
                )

            if errors:
                st.error("Dữ liệu nhập vào chưa hợp lệ.")

                for error in errors:
                    st.write(f"- {error}")

            else:
                recency = (
                    date.today() - last_purchase_date
                ).days

                with st.spinner(
                    "Đang chuẩn hóa dữ liệu và dự đoán phân khúc..."
                ):
                    time.sleep(0.5)

                    try:
                        result = predict_customer(
                            recency=recency,
                            frequency=int(frequency),
                            monetary=float(monetary),
                        )

                    except Exception as error:
                        st.error(
                            f"Không thể thực hiện dự đoán: {error}"
                        )
                        st.stop()

                cluster = int(
                    result.get(
                        "Cluster",
                        result.get("cluster", -1),
                    )
                )

                cluster_name = str(result.get("ClusterName", f"Cluster {cluster}"))

                cluster_description = result.get(
                    "ClusterDescription",
                    "",
                )

                cluster_content = get_cluster_content(cluster)
                with st.expander("Thông tin mô hình đang sử dụng"):
                    st.write("**Mô hình:**", result["ModelName"])
                    st.write("**Bộ chuẩn hóa:**", result["ScalerName"])
                    st.write("**File model:**", result["ModelPath"])

                # =================================================
                # LƯU LỊCH SỬ
                # =================================================
                single_history = pd.DataFrame(
                    [
                        {
                            "PredictionTime": (
                                datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ),
                            "PredictionType": "Single",
                            "CustomerID": customer_id.strip(),
                            "LastPurchaseDate": (
                                last_purchase_date.strftime(
                                    "%Y-%m-%d"
                                )
                            ),
                            "Recency": int(recency),
                            "Frequency": int(frequency),
                            "Monetary": float(monetary),
                            "Cluster": int(cluster),
                            "ClusterName": cluster_name,
                        }
                    ]
                )

                try:
                    append_prediction_history(single_history)

                except Exception as error:
                    st.warning(
                        "Dự đoán thành công nhưng chưa thể lưu lịch sử: "
                        f"{error}"
                    )

                # =================================================
                # THÔNG BÁO PHÂN KHÚC
                # =================================================
                message = cluster_name

                if cluster_content["message_type"] == "success":
                    st.success(message)

                elif cluster_content["message_type"] == "warning":
                    st.warning(message)

                elif cluster_content["message_type"] == "error":
                    st.error(message)

                else:
                    st.info(message)

                st.caption(
                    f"Mã khách hàng: {customer_id.strip()}"
                )

                # =================================================
                # CÁC CHỈ SỐ
                # =================================================
                metric_1, metric_2 = st.columns(2)

                metric_1.metric(
                    "Mã cụm",
                    f"Cluster {cluster}",
                )

                metric_2.metric(
                    "Recency",
                    f"{recency} ngày",
                )

                metric_3, metric_4 = st.columns(2)

                metric_3.metric(
                    "Frequency",
                    f"{int(frequency):,}",
                )

                metric_4.metric(
                    "Monetary",
                    f"{float(monetary):,.0f} ₫",
                )

                st.divider()

                # =================================================
                # TÓM TẮT KHÁCH HÀNG
                # =================================================
                st.subheader("Tóm tắt khách hàng")

                summary_col_1, summary_col_2 = st.columns(2)

                with summary_col_1:
                    st.write("**Mã khách hàng**")
                    st.write(customer_id.strip())

                with summary_col_2:
                    st.write("**Ngày mua gần nhất**")
                    st.write(
                        last_purchase_date.strftime("%d/%m/%Y")
                    )

                st.divider()

                # =================================================
                # ĐẶC ĐIỂM PHÂN KHÚC
                # =================================================
                st.subheader("Đặc điểm phân khúc")

                st.write(cluster_content["description"])

                # =================================================
                # ĐỀ XUẤT MARKETING
                # =================================================
                st.subheader("Đề xuất Marketing")

                for index, recommendation in enumerate(
                    cluster_content["recommendations"],
                    start=1,
                ):
                    with st.container(border=True):
                        st.write(
                            f"**{index}. {recommendation}**"
                        )

                st.caption(
                    "Kết quả được tạo từ mô hình K-Means đã huấn luyện "
                    "trên dữ liệu RFM."
                )