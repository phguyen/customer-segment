import io

import pandas as pd
import streamlit as st

from utils.prediction_utils import (
    APIConnectionError,
    APIResponseError,
    predict_customer_file,
)


# =========================================================
# 1. TIÊU ĐỀ
# =========================================================
st.title("Phân nhóm khách hàng hàng loạt")

st.caption(
    "Tải file giao dịch CSV để hệ thống phân nhóm hàng loạt."
)


# =========================================================
# 2. CẤU TRÚC FILE HỖ TRỢ
# =========================================================
with st.expander(
    "Cấu trúc file CSV được hỗ trợ",
    expanded=False,
):
    st.markdown(
        """
        File giao dịch cần có đủ năm nhóm thông tin:

        | Thông tin | Một số tên cột được hỗ trợ |
        |---|---|
        | Mã hóa đơn | `InvoiceNo`, `invoice_no`, `ma_hoa_don` |
        | Số lượng | `Quantity`, `quantity`, `so_luong`, `qty` |
        | Ngày hóa đơn | `InvoiceDate`, `invoice_date`, `ngay_mua` |
        | Đơn giá | `UnitPrice`, `unit_price`, `don_gia`, `price` |
        | Mã khách hàng | `CustomerID`, `customer_id`, `ma_khach_hang` |

        **Lưu ý:** Mỗi dòng trong file nên tương ứng với một sản phẩm
        thuộc một hóa đơn giao dịch.
        """
    )


# =========================================================
# 3. SESSION STATE
# =========================================================
if "batch_prediction_result" not in st.session_state:
    st.session_state.batch_prediction_result = None

if "batch_uploaded_file_name" not in st.session_state:
    st.session_state.batch_uploaded_file_name = None


# =========================================================
# 4. HÀM HỖ TRỢ
# =========================================================
def read_csv_preview(file_bytes: bytes) -> pd.DataFrame:
    """
    Đọc trước file CSV với một số encoding phổ biến.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1258",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                nrows=10,
                encoding=encoding,
            )

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "Không thể xác định bảng mã của file CSV."
    ) from last_error


def normalize_result_dataframe(
    result_rows: list,
) -> pd.DataFrame:
    """
    Chuyển dữ liệu API thành DataFrame và chuẩn hóa kiểu dữ liệu.
    """

    required_columns = [
        "customer_id",
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "cluster_id",
        "persona",
        "description",
        "confidence",
    ]

    result_df = pd.DataFrame(result_rows)

    for column in required_columns:
        if column not in result_df.columns:
            result_df[column] = None

    numeric_columns = [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "cluster_id",
    ]

    for column in numeric_columns:
        result_df[column] = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

    result_df["customer_id"] = (
        result_df["customer_id"]
        .fillna("")
        .astype(str)
    )

    result_df["persona"] = (
        result_df["persona"]
        .fillna("Chưa xác định")
        .astype(str)
    )

    result_df["description"] = (
        result_df["description"]
        .fillna("")
        .astype(str)
    )

    result_df["confidence"] = (
        result_df["confidence"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    return result_df[required_columns]


def translate_confidence(value: str) -> str:
    """
    Chuyển tên độ tin cậy sang tiếng Việt.
    """

    confidence_mapping = {
        "high": "Cao",
        "low": "Thấp",
        "unknown": "Chưa xác định",
    }

    return confidence_mapping.get(
        str(value).lower(),
        str(value),
    )


# =========================================================
# 5. TẢI FILE
# =========================================================
with st.container(border=True):
    st.subheader("Tải dữ liệu giao dịch")

    uploaded_file = st.file_uploader(
        "Chọn file giao dịch CSV",
        type=["csv"],
        help=(
            "File sẽ được gửi đến FastAPI để làm sạch dữ liệu, "
            "tính RFM và phân nhóm."
        ),
    )

    uploaded_file_bytes = None
    preview_df = None
    preview_error = None

    if uploaded_file is not None:
        uploaded_file_bytes = uploaded_file.getvalue()

        st.success(
            f"Đã chọn file: **{uploaded_file.name}**"
        )

        file_size_kb = len(uploaded_file_bytes) / 1024

        file_col_1, file_col_2 = st.columns(2)

        file_col_1.metric(
            "Tên file",
            uploaded_file.name,
        )

        file_col_2.metric(
            "Dung lượng",
            f"{file_size_kb:,.1f} KB",
        )

        try:
            preview_df = read_csv_preview(
                uploaded_file_bytes
            )

        except Exception as error:
            preview_error = str(error)

        if preview_df is not None:
            st.write("**Xem trước 10 dòng đầu tiên**")

            st.dataframe(
                preview_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"File hiện có {len(preview_df.columns)} cột. "
                "Bảng trên chỉ hiển thị tối đa 10 dòng đầu tiên."
            )

        else:
            st.warning(
                "Không thể xem trước nội dung file. "
                f"Chi tiết: {preview_error}"
            )

    button_col_1, button_col_2 = st.columns(
        [3, 1]
    )

    with button_col_1:
        submit_batch = st.button(
            "Phân tích file khách hàng",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    with button_col_2:
        clear_result = st.button(
            "Xóa kết quả",
            use_container_width=True,
            disabled=(
                st.session_state.batch_prediction_result
                is None
            ),
        )


# =========================================================
# 6. XÓA KẾT QUẢ CŨ
# =========================================================
if clear_result:
    st.session_state.batch_prediction_result = None
    st.session_state.batch_uploaded_file_name = None
    st.rerun()


# =========================================================
# 7. GỌI API
# =========================================================
if submit_batch and uploaded_file is not None:
    if not uploaded_file_bytes:
        st.error(
            "File được tải lên đang trống."
        )

    else:
        with st.spinner(
            "Backend đang làm sạch dữ liệu, tính RFM "
            "và phân nhóm khách hàng..."
        ):
            try:
                # Tạo lại đối tượng file từ bytes để tránh lỗi
                # con trỏ file đã bị thay đổi sau khi xem trước.
                api_file = io.BytesIO(
                    uploaded_file_bytes
                )

                api_file.name = uploaded_file.name

                result = predict_customer_file(
                    api_file
                )

                st.session_state.batch_prediction_result = {
                    "success": True,
                    "data": result,
                }

                st.session_state.batch_uploaded_file_name = (
                    uploaded_file.name
                )

            except APIConnectionError as error:
                st.session_state.batch_prediction_result = {
                    "success": False,
                    "error_type": "connection",
                    "error": str(error),
                }

            except APIResponseError as error:
                st.session_state.batch_prediction_result = {
                    "success": False,
                    "error_type": "response",
                    "error": str(error),
                }

            except Exception as error:
                st.session_state.batch_prediction_result = {
                    "success": False,
                    "error_type": "unknown",
                    "error": (
                        "Đã xảy ra lỗi không xác định: "
                        f"{error}"
                    ),
                }


# =========================================================
# 8. HIỂN THỊ KẾT QUẢ
# =========================================================
stored_result = (
    st.session_state.batch_prediction_result
)

if stored_result is None:
    st.info(
        "Chưa có kết quả phân nhóm hàng loạt. "
        "Hãy tải file CSV và nhấn nút phân tích."
    )

elif not stored_result.get("success"):
    error_type = stored_result.get(
        "error_type",
        "unknown",
    )

    error_message = stored_result.get(
        "error",
        "Không xác định được lỗi.",
    )

    if error_type == "connection":
        st.error(
            "Không thể kết nối đến backend."
        )

        st.info(
            "Hãy kiểm tra FastAPI đã được chạy bằng lệnh:\n\n"
            "`uvicorn backend.main:app --reload`"
        )

    elif error_type == "response":
        st.error(
            "Backend không thể xử lý file."
        )

    else:
        st.error(
            "Đã xảy ra lỗi trong quá trình phân tích."
        )

    st.code(
        error_message,
        language=None,
    )

else:
    response_data = stored_result.get(
        "data",
        {},
    )

    result_rows = response_data.get(
        "results",
        [],
    )

    total_from_api = int(
        response_data.get(
            "total_customers_segmented",
            len(result_rows),
        )
    )

    result_df = normalize_result_dataframe(
        result_rows
    )

    st.success(
        f"Đã phân nhóm thành công "
        f"**{total_from_api:,} khách hàng**."
    )

    if (
        st.session_state.batch_uploaded_file_name
        is not None
    ):
        st.caption(
            "Nguồn dữ liệu: "
            f"`{st.session_state.batch_uploaded_file_name}`"
        )

    if result_df.empty:
        st.warning(
            "Backend đã xử lý file nhưng không trả về "
            "khách hàng hợp lệ."
        )

    else:
        # =================================================
        # 9. KPI
        # =================================================
        total_customers = len(result_df)

        total_segments = (
            result_df["persona"]
            .dropna()
            .nunique()
        )

        high_confidence_count = (
            result_df["confidence"]
            .eq("high")
            .sum()
        )

        total_monetary = (
            result_df["monetary_value"]
            .fillna(0)
            .sum()
        )

        kpi_1, kpi_2, kpi_3 = (
            st.columns(3)
        )

        kpi_1.metric(
            "Tổng khách hàng",
            f"{total_customers:,}",
        )

        kpi_2.metric(
            "Số phân khúc",
            f"{total_segments:,}",
        )


        kpi_3.metric(
            "Tổng chi tiêu",
            f"{total_monetary:,.0f} ₫",
        )

        st.divider()

        # =================================================
        # 10. BIỂU ĐỒ TỔNG QUAN
        # =================================================
        chart_col_1, chart_col_2 = st.columns(
            2,
            gap="large",
        )

        with chart_col_1:
            with st.container(border=True):
                st.subheader(
                    "Phân bố khách hàng theo phân khúc"
                )

                persona_distribution = (
                    result_df.groupby(
                        "persona"
                    )
                    .size()
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    persona_distribution,
                    use_container_width=True,
                )

        with chart_col_2:
            with st.container(border=True):
                st.subheader(
                    "Monetary trung bình theo phân khúc"
                )

                monetary_by_persona = (
                    result_df.groupby(
                        "persona"
                    )["monetary_value"]
                    .mean()
                    .fillna(0)
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    monetary_by_persona,
                    use_container_width=True,
                )

        # =================================================
        # 11. THỐNG KÊ RFM THEO PHÂN KHÚC
        # =================================================
        with st.container(border=True):
            st.subheader(
                "RFM trung bình theo phân khúc"
            )

            rfm_summary = (
                result_df.groupby(
                    [
                        "cluster_id",
                        "persona",
                    ],
                    dropna=False,
                )
                .agg(
                    SoKhachHang=(
                        "customer_id",
                        "count",
                    ),
                    RecencyTrungBinh=(
                        "recency_days",
                        "mean",
                    ),
                    FrequencyTrungBinh=(
                        "frequency_orders",
                        "mean",
                    ),
                    MonetaryTrungBinh=(
                        "monetary_value",
                        "mean",
                    ),
                )
                .reset_index()
            )

            rfm_summary["cluster_id"] = (
                rfm_summary["cluster_id"]
                .fillna(-1)
                .astype(int)
            )

            st.dataframe(
                rfm_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "cluster_id": (
                        st.column_config.NumberColumn(
                            "Mã cụm",
                            format="%d",
                        )
                    ),
                    "persona": (
                        st.column_config.TextColumn(
                            "Phân khúc"
                        )
                    ),
                    "SoKhachHang": (
                        st.column_config.NumberColumn(
                            "Số khách hàng",
                            format="%d",
                        )
                    ),
                    "RecencyTrungBinh": (
                        st.column_config.NumberColumn(
                            "Recency trung bình",
                            format="%.1f ngày",
                        )
                    ),
                    "FrequencyTrungBinh": (
                        st.column_config.NumberColumn(
                            "Frequency trung bình",
                            format="%.1f",
                        )
                    ),
                    "MonetaryTrungBinh": (
                        st.column_config.NumberColumn(
                            "Monetary trung bình",
                            format="%.0f ₫",
                        )
                    ),
                },
            )

        # =================================================
        # 12. BỘ LỌC
        # =================================================
        st.subheader("Bộ lọc kết quả")

        with st.container(border=True):
            filter_col_1, filter_col_2, filter_col_3 = (
                st.columns(3)
            )

            persona_options = sorted(
                result_df["persona"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_personas = (
                filter_col_1.multiselect(
                    "Phân khúc khách hàng",
                    options=persona_options,
                    default=persona_options,
                )
            )

            confidence_options = sorted(
                result_df["confidence"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_confidence = (
                filter_col_2.multiselect(
                    "Độ tin cậy",
                    options=confidence_options,
                    default=confidence_options,
                    format_func=translate_confidence,
                )
            )

            customer_search = (
                filter_col_3.text_input(
                    "Tìm mã khách hàng",
                    placeholder="Nhập mã khách hàng",
                )
            )

        filtered_df = result_df.copy()

        if selected_personas:
            filtered_df = filtered_df[
                filtered_df["persona"].isin(
                    selected_personas
                )
            ]

        else:
            filtered_df = filtered_df.iloc[0:0]

        if selected_confidence:
            filtered_df = filtered_df[
                filtered_df["confidence"].isin(
                    selected_confidence
                )
            ]

        else:
            filtered_df = filtered_df.iloc[0:0]

        if customer_search.strip():
            filtered_df = filtered_df[
                filtered_df["customer_id"].str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False,
                )
            ]

        st.caption(
            f"Đang hiển thị {len(filtered_df):,}/"
            f"{len(result_df):,} khách hàng."
        )

        # =================================================
        # 13. DANH SÁCH KẾT QUẢ
        # =================================================
        st.subheader("Danh sách khách hàng")

        display_columns = [
            "customer_id",
            "recency_days",
            "frequency_orders",
            "monetary_value",
            "cluster_id",
            "persona",
            "description",
            "confidence",
        ]

        display_df = filtered_df[
            display_columns
        ].copy()

        display_df["confidence"] = (
            display_df["confidence"]
            .apply(translate_confidence)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "customer_id": (
                    st.column_config.TextColumn(
                        "Mã khách hàng"
                    )
                ),
                "recency_days": (
                    st.column_config.NumberColumn(
                        "Lần cuối mua",
                        format="%.0f ngày",
                    )
                ),
                "frequency_orders": (
                    st.column_config.NumberColumn(
                        "Số lần mua",
                        format="%.0f",
                    )
                ),
                "monetary_value": (
                    st.column_config.NumberColumn(
                        "Tổng chi tiêu",
                        format="%.0f $",
                    )
                ),
                "cluster_id": (
                    st.column_config.NumberColumn(
                        "Mã cụm",
                        format="%.0f",
                    )
                ),
                "persona": (
                    st.column_config.TextColumn(
                        "Phân khúc"
                    )
                ),
                "description": (
                    st.column_config.TextColumn(
                        "Mô tả",
                        width="large",
                    )
                ),
                "confidence": (
                    st.column_config.TextColumn(
                        "Độ tin cậy"
                    )
                ),
            },
        )

        # =================================================
        # 14. TẢI KẾT QUẢ
        # =================================================
        csv_data = display_df.to_csv(
            index=False,
        ).encode("utf-8-sig")

        st.download_button(
            label="Tải kết quả phân nhóm (.CSV)",
            data=csv_data,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            disabled=display_df.empty,
        )
