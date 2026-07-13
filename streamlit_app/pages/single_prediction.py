from datetime import date
import time

import streamlit as st

from datetime import datetime
import pandas as pd

from utils.history_utils import (
    append_prediction_history,
)
# =========================================================
# 1. HÀM DỰ ĐOÁN DEMO
# =========================================================
def predict_customer(recency, frequency, monetary):
    """
    Logic demo để kiểm tra giao diện.

    Sau này khi có model thật, thay phần này bằng:
    - scaler.transform(...)
    - model.predict(...)
    """

    if recency <= 30 and frequency >= 8 and monetary >= 5_000_000:
        return {
            "cluster": 0,
            "name": "Khách hàng giá trị cao",
            "status": "VIP",
            "description": (
                "Khách hàng mua gần đây, có tần suất mua cao "
                "và tổng giá trị chi tiêu lớn."
            ),
            "recommendations": [
                "Ưu tiên chương trình khách hàng VIP",
                "Cung cấp ưu đãi độc quyền",
                "Cá nhân hóa nội dung chăm sóc",
                "Khuyến khích giới thiệu khách hàng mới",
            ],
        }

    if recency <= 90 and frequency >= 4 and monetary >= 1_500_000:
        return {
            "cluster": 1,
            "name": "Khách hàng tiềm năng",
            "status": "Potential",
            "description": (
                "Khách hàng có mức độ tương tác tốt "
                "và còn khả năng gia tăng giá trị."
            ),
            "recommendations": [
                "Khuyến khích mua lại bằng voucher",
                "Đề xuất combo sản phẩm phù hợp",
                "Gửi nội dung cá nhân hóa",
                "Theo dõi phản hồi sau mua",
            ],
        }

    if recency > 180 or frequency <= 1:
        return {
            "cluster": 3,
            "name": "Khách hàng có nguy cơ rời bỏ",
            "status": "At Risk",
            "description": (
                "Khách hàng đã lâu chưa quay lại "
                "hoặc có hoạt động mua hàng rất thấp."
            ),
            "recommendations": [
                "Triển khai chiến dịch tái kích hoạt",
                "Gửi ưu đãi quay lại",
                "Khảo sát nguyên nhân ngừng mua",
                "Ưu tiên chăm sóc trong 7 ngày tới",
            ],
        }

    return {
        "cluster": 2,
        "name": "Khách hàng cần chăm sóc",
        "status": "Need Attention",
        "description": (
            "Khách hàng có mức độ mua hàng trung bình "
            "và cần được nuôi dưỡng thêm."
        ),
        "recommendations": [
            "Gửi nội dung nhắc nhớ thương hiệu",
            "Áp dụng ưu đãi nhẹ",
            "Gợi ý sản phẩm phù hợp",
            "Theo dõi tần suất mua trong 30 ngày",
        ],
    }


# =========================================================
# 2. CSS NHẸ, KHÔNG NHÉT HTML PHỨC TẠP
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
# 3. TIÊU ĐỀ TRANG
# =========================================================
st.title("Phân nhóm khách hàng đơn lẻ")

st.caption(
    "Kiểm tra nhanh phân khúc khách hàng dựa trên hành vi mua hàng RFM."
)

st.info(
    "Demo Mode: Màn hình hiện đang sử dụng logic minh họa, "
    "chưa kết nối với mô hình Machine Learning thật."
)


# =========================================================
# 4. BỐ CỤC HAI CỘT
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
            "Nhập các chỉ số hành vi mua hàng để hệ thống thực hiện phân tích."
        )

        with st.form("single_prediction_form"):
            customer_id = st.text_input(
                "Mã khách hàng",
                placeholder="Ví dụ: CUST001",
                help="Mã định danh khách hàng trong hệ thống CRM.",
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
# 6. KHU VỰC KẾT QUẢ
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

            placeholder_1.metric(
                "Mã cụm",
                "—",
            )

            placeholder_2.metric(
                "Recency",
                "—",
            )

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
                    "Đang phân tích hành vi khách hàng..."
                ):
                    time.sleep(0.8)

                result = predict_customer(
                    recency=recency,
                    frequency=int(frequency),
                    monetary=float(monetary),
                )
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
                            "Recency": recency,
                            "Frequency": int(frequency),
                            "Monetary": float(monetary),
                            "Cluster": result["cluster"],
                            "ClusterName": result["name"],
                        }
                    ]
                )

                append_prediction_history(
                    single_history
                )

                # =================================================
                # THÔNG BÁO THEO TỪNG LOẠI CỤM
                # =================================================
                if result["cluster"] == 0:
                    st.success(
                        f"{result['name']} · {result['status']}"
                    )

                elif result["cluster"] == 1:
                    st.info(
                        f"{result['name']} · {result['status']}"
                    )

                elif result["cluster"] == 2:
                    st.warning(
                        f"{result['name']} · {result['status']}"
                    )

                else:
                    st.error(
                        f"{result['name']} · {result['status']}"
                    )

                st.caption(
                    f"Mã khách hàng: {customer_id.strip()}"
                )

                # =================================================
                # CÁC CHỈ SỐ CHÍNH
                # =================================================
                metric_1, metric_2 = st.columns(2)

                metric_1.metric(
                    "Mã cụm",
                    f"Cluster {result['cluster']}",
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
                # THÔNG TIN KHÁCH HÀNG
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

                st.write(result["description"])

                # =================================================
                # ĐỀ XUẤT MARKETING
                # =================================================
                st.subheader("Đề xuất Marketing")

                for index, recommendation in enumerate(
                    result["recommendations"],
                    start=1,
                ):
                    with st.container(border=True):
                        st.write(
                            f"**{index}. {recommendation}**"
                        )

                st.caption(
                    "Kết quả hiện được tạo bằng logic minh họa, "
                    "chưa phải đầu ra từ mô hình Machine Learning."
                )