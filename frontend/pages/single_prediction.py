from datetime import date

import streamlit as st

from utils.prediction_utils import (
    APIConnectionError,
    APIResponseError,
    is_backend_running,
    predict_customer,
)

# =========================================================
# 1. NỘI DUNG MARKETING THEO TỪNG CỤM
# =========================================================
CLUSTER_CONTENT = {
    0: {
        "description": (
            "Đây là nhóm khách hàng tiềm năng mới gia nhập, "
            "cần được khuyến khích mua thêm và duy trì tương tác."
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
            "Đây là nhóm khách hàng có giá trị cao, "
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


# =========================================================
# 2. HÀM HỖ TRỢ
# =========================================================
def get_cluster_content(cluster_id: int) -> dict:
    """
    Trả về mô tả và đề xuất marketing theo mã cụm.
    """
    return CLUSTER_CONTENT.get(
        cluster_id,
        {
            "description": (
                "Khách hàng đã được phân nhóm dựa trên ba chỉ số "
                "Recency, Frequency và Monetary."
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


def get_model_value(
    model_info: dict,
    possible_keys: list[str],
    default="—",
):
    """
    Lấy giá trị từ thông tin model khi backend có thể sử dụng
    tên khóa tiếng Anh hoặc tiếng Việt khác nhau.
    """
    for key in possible_keys:
        value = model_info.get(key)

        if value is not None:
            return value

    return default


def show_persona_message(
    persona: str,
    message_type: str,
):
    """
    Hiển thị tên phân khúc theo kiểu thông báo phù hợp.
    """
    if message_type == "success":
        st.success(persona)

    elif message_type == "warning":
        st.warning(persona)

    elif message_type == "error":
        st.error(persona)

    else:
        st.info(persona)


# =========================================================
# 3. CSS
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
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. TIÊU ĐỀ
# =========================================================
st.title("Phân nhóm khách hàng đơn lẻ")

st.caption(
    "Nhập thông tin từng khách hàng để phân tích"
)


# =========================================================
# 5. KIỂM TRA BACKEND
# =========================================================
if not is_backend_running():
    st.error(
        "Không thể kết nối tới FastAPI Backend. "
        "Hãy chạy backend trước khi thực hiện dự đoán."
    )

    st.code(
        "uvicorn main:app --reload",
        language="bash",
    )

    st.stop()


# =========================================================
# 6. SESSION STATE
# =========================================================
if "single_prediction_result" not in st.session_state:
    st.session_state.single_prediction_result = None

if "single_prediction_input" not in st.session_state:
    st.session_state.single_prediction_input = None


# =========================================================
# 7. BỐ CỤC
# =========================================================
input_col, result_col = st.columns(
    [1.02, 0.98],
    gap="large",
)


# =========================================================
# 8. FORM NHẬP
# =========================================================
with input_col:
    with st.container(border=True):
        st.subheader("Thông tin khách hàng")

        st.caption(
            "Nhập thông tin khách hàng để dự đoán phân khúc."
        )

        with st.form(
            "single_prediction_form",
            clear_on_submit=False,
        ):
            customer_id = st.text_input(
                "Mã khách hàng",
                placeholder="Ví dụ: CHI_AN",
                help=(
                    "Dùng để định danh và lưu trữ kết quả dự đoán của khách hàng."
                ),
            )

            last_purchase_date = st.date_input(
                "Ngày mua hàng gần nhất",
                value=date.today(),
                max_value=date.today(),
                format="DD/MM/YYYY",
            )

            frequency = st.number_input(
                "Tổng số lần mua hàng",
                min_value=1,
                value=1,
                step=1,
                help="Số lần khách hàng đã thực hiện giao dịch.",
            )

            monetary = st.number_input(
                "Tổng số tiền đã chi tiêu",
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
# 9. XỬ LÝ DỰ ĐOÁN
# =========================================================
if submitted:
    validation_errors = []

    clean_customer_id = customer_id.strip()

    if not clean_customer_id:
        validation_errors.append(
            "Vui lòng nhập mã khách hàng."
        )

    if frequency < 1:
        validation_errors.append(
            "Frequency phải lớn hơn hoặc bằng 1."
        )

    if monetary <= 0:
        validation_errors.append(
            "Monetary phải lớn hơn 0."
        )

    if last_purchase_date > date.today():
        validation_errors.append(
            "Ngày mua gần nhất không được lớn hơn ngày hiện tại."
        )

    if validation_errors:
        st.session_state.single_prediction_result = {
            "validation_errors": validation_errors
        }

        st.session_state.single_prediction_input = None

    else:
        with st.spinner(
            "Đang gửi dữ liệu đến FastAPI và dự đoán phân khúc..."
        ):
            try:
                api_result = predict_customer(
                    customer_id=clean_customer_id,
                    last_purchase_date=last_purchase_date,
                    frequency=frequency,
                    monetary=monetary,
                )

                recency = (
                    date.today() - last_purchase_date
                ).days

                st.session_state.single_prediction_result = (
                    api_result
                )

                st.session_state.single_prediction_input = {
                    "customer_id": clean_customer_id,
                    "last_purchase_date": last_purchase_date,
                    "recency": int(recency),
                    "frequency": int(frequency),
                    "monetary": float(monetary),
                }

            except (
                APIConnectionError,
                APIResponseError,
            ) as error:
                st.session_state.single_prediction_result = {
                    "api_error": str(error)
                }

                st.session_state.single_prediction_input = None

            except Exception as error:
                st.session_state.single_prediction_result = {
                    "api_error": (
                        "Đã xảy ra lỗi không xác định: "
                        f"{error}"
                    )
                }

                st.session_state.single_prediction_input = None


# =========================================================
# 10. KẾT QUẢ
# =========================================================
with result_col:
    with st.container(border=True):
        st.subheader("Kết quả phân tích")

        result = st.session_state.single_prediction_result
        input_data = st.session_state.single_prediction_input

        if result is None:
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
                "Độ tin cậy",
                "—",
            )

            st.divider()

            st.subheader("Đề xuất Marketing")

            st.write(
                "Đề xuất sẽ được hiển thị sau khi hệ thống "
                "xác định được phân khúc khách hàng."
            )

        elif "validation_errors" in result:
            st.error(
                "Dữ liệu nhập vào chưa hợp lệ."
            )

            for validation_error in result[
                "validation_errors"
            ]:
                st.write(
                    f"- {validation_error}"
                )

        elif "api_error" in result:
            st.error(
                "Không thể thực hiện dự đoán."
            )

            st.write(
                result["api_error"]
            )

        else:
            try:
                cluster_id = int(
                    result.get(
                        "cluster_id",
                        result.get(
                            "cluster",
                            -1,
                        ),
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                cluster_id = -1

            persona = str(
                result.get(
                    "persona",
                    result.get(
                        "cluster_label",
                        f"Nhóm khách hàng {cluster_id}",
                    ),
                )
            )

            api_description = str(
                result.get(
                    "description",
                    "",
                )
                or ""
            )

            confidence = str(
                result.get(
                    "confidence",
                    "unknown",
                )
                or "unknown"
            ).lower()

            cluster_content = get_cluster_content(
                cluster_id
            )

            show_persona_message(
                persona=persona,
                message_type=cluster_content[
                    "message_type"
                ],
            )

            if input_data:
                st.caption(
                    f"Mã khách hàng: "
                    f"{input_data['customer_id']}"
                )

            metric_1, metric_2 = st.columns(2)

            metric_1.metric(
                "Mã cụm",
                (
                    f"Cluster {cluster_id}"
                    if cluster_id >= 0
                    else "Chưa xác định"
                ),
            )

            confidence_display = {
                "high": "Cao",
                "medium": "Trung bình",
                "low": "Thấp",
                "unknown": "Chưa xác định",
            }.get(
                confidence,
                confidence.capitalize(),
            )

            metric_2.metric(
                "Độ tin cậy",
                confidence_display,
            )

            metric_3, metric_4, metric_5 = st.columns(3)

            metric_3.metric(
                "Recency",
                (
                    f"{input_data['recency']} ngày"
                    if input_data
                    else "—"
                ),
            )

            metric_4.metric(
                "Frequency",
                (
                    f"{input_data['frequency']:,}"
                    if input_data
                    else "—"
                ),
            )

            metric_5.metric(
                "Monetary",
                (
                    f"{input_data['monetary']:,.0f} ₫"
                    if input_data
                    else "—"
                ),
            )

            st.divider()

            if input_data:
                st.subheader(
                    "Tóm tắt khách hàng"
                )

                summary_col_1, summary_col_2 = st.columns(2)

                with summary_col_1:
                    st.write(
                        "**Mã khách hàng**"
                    )

                    st.write(
                        input_data[
                            "customer_id"
                        ]
                    )

                with summary_col_2:
                    st.write(
                        "**Ngày mua gần nhất**"
                    )

                    st.write(
                        input_data[
                            "last_purchase_date"
                        ].strftime(
                            "%d/%m/%Y"
                        )
                    )

            st.divider()

            st.subheader(
                "Đặc điểm phân khúc"
            )

            description = (
                api_description.strip()
                or cluster_content[
                    "description"
                ]
            )

            st.write(description)

            st.subheader(
                "Đề xuất Marketing"
            )

            for index, recommendation in enumerate(
                cluster_content[
                    "recommendations"
                ],
                start=1,
            ):
                with st.container(
                    border=True
                ):
                    st.write(
                        f"**{index}. "
                        f"{recommendation}**"
                    )

            