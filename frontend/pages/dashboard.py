import pandas as pd
import streamlit as st

from utils.api_client import get_prediction_history


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.title("Tổng quan khách hàng")

st.caption(
    "Theo dõi số lượt phân tích, giá trị RFM và "
    "phân bố các nhóm khách hàng từ dữ liệu API."
)


# =========================================================
# 2. HÀM HỖ TRỢ
# =========================================================
def format_currency(value):
    """
    Định dạng giá trị tiền tệ.
    """

    try:
        return f"{float(value):,.0f} ₫"
    except (TypeError, ValueError):
        return "0 ₫"


def prepare_history_dataframe(history_data):
    """
    Chuyển dữ liệu JSON từ API thành DataFrame
    và chuẩn hóa tên cột cho frontend.
    """

    if not history_data:
        return pd.DataFrame(
            columns=[
                "ID",
                "PredictionTime",
                "Recency",
                "Frequency",
                "Monetary",
                "Cluster",
                "ClusterName"
            ]
        )

    history_df = pd.DataFrame(history_data)

    api_column_mapping = {
        "id": "ID",
        "created_at": "PredictionTime",
        "recency": "Recency",
        "frequency": "Frequency",
        "monetary": "Monetary",
        "cluster_id": "Cluster",
        "cluster_label": "ClusterName"
    }

    history_df = history_df.rename(
        columns=api_column_mapping
    )

    required_columns = [
        "ID",
        "PredictionTime",
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
        "ClusterName"
    ]

    for column in required_columns:
        if column not in history_df.columns:
            history_df[column] = None

    history_df["PredictionTime"] = pd.to_datetime(
        history_df["PredictionTime"],
        errors="coerce"
    )

    numeric_columns = [
        "ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster"
    ]

    for column in numeric_columns:
        history_df[column] = pd.to_numeric(
            history_df[column],
            errors="coerce"
        )

    history_df["ClusterName"] = (
        history_df["ClusterName"]
        .fillna("Chưa xác định")
        .astype(str)
    )

    return history_df[required_columns]


# =========================================================
# 3. GỌI API
# =========================================================
with st.spinner("Đang tải dữ liệu từ hệ thống..."):
    api_result = get_prediction_history()


if not api_result["success"]:
    st.error(api_result["error"])

    st.info(
        "Hãy chạy backend bằng lệnh:\n\n"
        "`uvicorn backend.main:app --reload`"
    )

    st.stop()


history_df = prepare_history_dataframe(
    api_result["data"]
)


# =========================================================
# 4. TRƯỜNG HỢP CHƯA CÓ DỮ LIỆU
# =========================================================
if history_df.empty:
    st.info(
        "Chưa có lịch sử phân tích. Hãy thực hiện dự đoán tại "
        "trang Phân nhóm đơn lẻ hoặc Phân nhóm hàng loạt."
    )

    placeholder_1, placeholder_2, placeholder_3, placeholder_4 = (
        st.columns(4)
    )

    placeholder_1.metric(
        "Lượt phân tích",
        "0"
    )

    placeholder_2.metric(
        "Phân khúc",
        "0"
    )

    placeholder_3.metric(
        "Frequency trung bình",
        "0"
    )

    placeholder_4.metric(
        "Tổng Monetary",
        "0 ₫"
    )

    st.stop()


# =========================================================
# 5. BỘ LỌC
# =========================================================
with st.container(border=True):
    st.subheader("Bộ lọc dữ liệu")

    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

    available_clusters = sorted(
        history_df["ClusterName"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_clusters = filter_col_1.multiselect(
        "Phân khúc khách hàng",
        options=available_clusters,
        default=available_clusters
    )

    valid_dates = history_df[
        "PredictionTime"
    ].dropna()

    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_date_range = filter_col_2.date_input(
            "Khoảng thời gian",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    else:
        selected_date_range = None

        filter_col_2.info(
            "Chưa có dữ liệu thời gian."
        )

    search_cluster = filter_col_3.text_input(
        "Tìm tên phân khúc",
        placeholder="Ví dụ: VIP"
    )


# =========================================================
# 6. ÁP DỤNG BỘ LỌC
# =========================================================
filtered_df = history_df.copy()


if selected_clusters:
    filtered_df = filtered_df[
        filtered_df["ClusterName"].isin(
            selected_clusters
        )
    ]


if (
    selected_date_range
    and isinstance(selected_date_range, tuple)
    and len(selected_date_range) == 2
):
    start_date, end_date = selected_date_range

    filtered_df = filtered_df[
        filtered_df["PredictionTime"]
        .dt.date
        .between(
            start_date,
            end_date
        )
    ]


if search_cluster.strip():
    filtered_df = filtered_df[
        filtered_df["ClusterName"].str.contains(
            search_cluster.strip(),
            case=False,
            na=False
        )
    ]


# =========================================================
# 7. KPI TỔNG QUAN
# =========================================================
total_predictions = len(filtered_df)

total_clusters = (
    filtered_df["Cluster"]
    .dropna()
    .nunique()
)

average_frequency = (
    filtered_df["Frequency"]
    .fillna(0)
    .mean()
)

total_monetary = (
    filtered_df["Monetary"]
    .fillna(0)
    .sum()
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Lượt phân tích",
    f"{total_predictions:,}"
)

metric_2.metric(
    "Phân khúc",
    f"{total_clusters:,}"
)

metric_3.metric(
    "Frequency trung bình",
    f"{average_frequency:,.1f}"
)

metric_4.metric(
    "Tổng Monetary",
    format_currency(total_monetary)
)


st.divider()


# =========================================================
# 8. BIỂU ĐỒ PHÂN BỐ PHÂN KHÚC
# =========================================================
chart_col_1, chart_col_2 = st.columns(
    2,
    gap="large"
)


with chart_col_1:
    with st.container(border=True):
        st.subheader(
            "Số lượt phân tích theo phân khúc"
        )

        cluster_distribution = (
            filtered_df.groupby(
                "ClusterName",
                dropna=False
            )
            .size()
            .reset_index(
                name="Số lượt phân tích"
            )
            .sort_values(
                "Số lượt phân tích",
                ascending=False
            )
        )

        if cluster_distribution.empty:
            st.info(
                "Không có dữ liệu phù hợp với bộ lọc."
            )

        else:
            st.bar_chart(
                cluster_distribution.set_index(
                    "ClusterName"
                )["Số lượt phân tích"],
                use_container_width=True
            )


with chart_col_2:
    with st.container(border=True):
        st.subheader(
            "Monetary trung bình theo phân khúc"
        )

        monetary_by_cluster = (
            filtered_df.groupby(
                "ClusterName",
                dropna=False
            )["Monetary"]
            .mean()
            .fillna(0)
            .sort_values(
                ascending=False
            )
        )

        if monetary_by_cluster.empty:
            st.info(
                "Không có dữ liệu phù hợp với bộ lọc."
            )

        else:
            st.bar_chart(
                monetary_by_cluster,
                use_container_width=True
            )


# =========================================================
# 9. XU HƯỚNG PHÂN TÍCH THEO THỜI GIAN
# =========================================================
with st.container(border=True):
    st.subheader(
        "Số lượt phân tích theo thời gian"
    )

    timeline_df = filtered_df.dropna(
        subset=["PredictionTime"]
    ).copy()

    if timeline_df.empty:
        st.info(
            "Chưa có dữ liệu thời gian để hiển thị."
        )

    else:
        timeline_df["PredictionDate"] = (
            timeline_df["PredictionTime"]
            .dt.date
        )

        daily_predictions = (
            timeline_df.groupby(
                "PredictionDate"
            )
            .size()
            .reset_index(
                name="Số lượt phân tích"
            )
            .set_index(
                "PredictionDate"
            )
        )

        st.line_chart(
            daily_predictions,
            use_container_width=True
        )


# =========================================================
# 10. RFM TRUNG BÌNH THEO PHÂN KHÚC
# =========================================================
with st.container(border=True):
    st.subheader(
        "Chỉ số RFM trung bình theo phân khúc"
    )

    rfm_summary = (
        filtered_df.groupby(
            [
                "Cluster",
                "ClusterName"
            ],
            dropna=False
        )[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean")
        )
        .reset_index()
    )

    if rfm_summary.empty:
        st.info(
            "Không có dữ liệu RFM phù hợp."
        )

    else:
        rfm_summary["Cluster"] = (
            rfm_summary["Cluster"]
            .fillna(-1)
            .astype(int)
        )

        rfm_summary["Recency"] = (
            rfm_summary["Recency"]
            .fillna(0)
            .round(1)
        )

        rfm_summary["Frequency"] = (
            rfm_summary["Frequency"]
            .fillna(0)
            .round(1)
        )

        rfm_summary["Monetary"] = (
            rfm_summary["Monetary"]
            .fillna(0)
            .round(0)
        )

        st.dataframe(
            rfm_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cluster": st.column_config.NumberColumn(
                    "Mã cụm",
                    format="%d"
                ),
                "ClusterName": st.column_config.TextColumn(
                    "Tên phân khúc"
                ),
                "Recency": st.column_config.NumberColumn(
                    "Recency trung bình",
                    format="%.1f ngày"
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency trung bình",
                    format="%.1f"
                ),
                "Monetary": st.column_config.NumberColumn(
                    "Monetary trung bình",
                    format="%.0f ₫"
                )
            }
        )


# =========================================================
# 11. THỐNG KÊ CHI TIẾT TỪNG PHÂN KHÚC
# =========================================================
with st.container(border=True):
    st.subheader(
        "Thống kê chi tiết theo phân khúc"
    )

    cluster_summary = (
        filtered_df.groupby(
            [
                "Cluster",
                "ClusterName"
            ],
            dropna=False
        )
        .agg(
            SoLuotPhanTich=(
                "ID",
                "count"
            ),
            RecencyThapNhat=(
                "Recency",
                "min"
            ),
            RecencyCaoNhat=(
                "Recency",
                "max"
            ),
            TongMonetary=(
                "Monetary",
                "sum"
            )
        )
        .reset_index()
    )

    if cluster_summary.empty:
        st.info(
            "Không có dữ liệu thống kê phù hợp."
        )

    else:
        cluster_summary["Cluster"] = (
            cluster_summary["Cluster"]
            .fillna(-1)
            .astype(int)
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cluster": st.column_config.NumberColumn(
                    "Mã cụm",
                    format="%d"
                ),
                "ClusterName": "Tên phân khúc",
                "SoLuotPhanTich": st.column_config.NumberColumn(
                    "Số lượt phân tích",
                    format="%d"
                ),
                "RecencyThapNhat": st.column_config.NumberColumn(
                    "Recency thấp nhất",
                    format="%.0f ngày"
                ),
                "RecencyCaoNhat": st.column_config.NumberColumn(
                    "Recency cao nhất",
                    format="%.0f ngày"
                ),
                "TongMonetary": st.column_config.NumberColumn(
                    "Tổng Monetary",
                    format="%.0f ₫"
                )
            }
        )


# =========================================================
# 12. LỊCH SỬ PHÂN TÍCH GẦN NHẤT
# =========================================================
with st.container(border=True):
    st.subheader(
        "Lịch sử phân tích gần nhất"
    )

    recent_df = (
        filtered_df.sort_values(
            "PredictionTime",
            ascending=False
        )
        .head(10)
        .copy()
    )

    if recent_df.empty:
        st.info(
            "Không có dữ liệu lịch sử phù hợp."
        )

    else:
        recent_df["PredictionTime"] = (
            recent_df["PredictionTime"]
            .dt.strftime(
                "%d/%m/%Y %H:%M"
            )
            .fillna("")
        )

        recent_df["ID"] = (
            recent_df["ID"]
            .fillna(-1)
            .astype(int)
        )

        recent_df["Cluster"] = (
            recent_df["Cluster"]
            .fillna(-1)
            .astype(int)
        )

        recent_df["Recency"] = (
            recent_df["Recency"]
            .fillna(0)
            .round(0)
            .astype(int)
        )

        recent_df["Frequency"] = (
            recent_df["Frequency"]
            .fillna(0)
            .round(0)
            .astype(int)
        )

        recent_df["Monetary"] = (
            recent_df["Monetary"]
            .fillna(0)
            .astype(float)
        )

        display_columns = [
            "ID",
            "PredictionTime",
            "Recency",
            "Frequency",
            "Monetary",
            "Cluster",
            "ClusterName"
        ]

        st.dataframe(
            recent_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn(
                    "Mã lịch sử",
                    format="%d"
                ),
                "PredictionTime": "Thời gian",
                "Recency": st.column_config.NumberColumn(
                    "Recency",
                    format="%d ngày"
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency",
                    format="%d"
                ),
                "Monetary": st.column_config.NumberColumn(
                    "Monetary",
                    format="%.0f ₫"
                ),
                "Cluster": st.column_config.NumberColumn(
                    "Mã cụm",
                    format="%d"
                ),
                "ClusterName": "Tên phân khúc"
            }
        )


# =========================================================
# 13. KIỂM TRA DỮ LIỆU API
# =========================================================
with st.expander(
    "Xem dữ liệu API",
    expanded=False
):
    st.json(api_result["data"])