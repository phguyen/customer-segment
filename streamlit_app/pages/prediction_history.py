from datetime import date

import pandas as pd
import streamlit as st

from utils.history_utils import (
    clear_prediction_history,
    load_prediction_history,
)


# =========================================================
# 1. TIÊU ĐỀ
# =========================================================
st.title("Lịch sử phân tích dữ liệu")

st.caption(
    "Tra cứu lịch sử phân nhóm khách hàng "
    "theo mã khách hàng, thời gian và cụm."
)


# =========================================================
# 2. ĐỌC LỊCH SỬ
# =========================================================
history_df = load_prediction_history()


if history_df.empty:
    st.info(
        "Chưa có lịch sử phân tích. "
        "Hãy thực hiện phân nhóm đơn lẻ "
        "hoặc phân nhóm hàng loạt trước."
    )

    st.stop()


# =========================================================
# 3. KPI TỔNG QUAN
# =========================================================
total_records = len(history_df)

total_customers = (
    history_df["CustomerID"]
    .dropna()
    .nunique()
)

total_clusters = (
    history_df["Cluster"]
    .dropna()
    .nunique()
)


kpi_col_1, kpi_col_2, kpi_col_3 = st.columns(3)

kpi_col_1.metric(
    "Tổng bản ghi",
    f"{total_records:,}",
)

kpi_col_2.metric(
    "Tổng khách hàng",
    f"{total_customers:,}",
)

# kpi_col_3.metric(
#     "Số cụm",
#     f"{total_clusters:,}",
# )
kpi_col_3.metric(
    "Số phân khúc",
    f"{total_clusters:,}",
)


# =========================================================
# 4. SEARCH VÀ FILTER
# =========================================================
st.subheader("Tìm kiếm và bộ lọc")

filter_col_1, filter_col_2, filter_col_3 = st.columns(
    [0.9, 1.5, 0.9],
    gap="medium",
)


with filter_col_1:
    search_keyword = st.text_input(
        "Tìm kiếm mã khách hàng",
        placeholder="Ví dụ: CUST001",
    )


with filter_col_2:
    # cluster_options = sorted(
    #     history_df["Cluster"]
    #     .dropna()
    #     .astype(int)
    #     .unique()
    #     .tolist()
    # )

    # selected_clusters = st.multiselect(
    #     "Lọc theo cụm",
    #     options=cluster_options,
    #     default=cluster_options,
    #     format_func=lambda value: f"Cluster {value}",
    # )
    cluster_name_options = sorted(
        history_df["ClusterName"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_cluster_names = st.multiselect(
        "Lọc theo phân khúc",
        options=cluster_name_options,
        default=cluster_name_options,
    )


with filter_col_3:
    prediction_type_options = sorted(
        history_df["PredictionType"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_prediction_types = st.multiselect(
        "Loại phân tích",
        options=prediction_type_options,
        default=prediction_type_options,
    )


# =========================================================
# 5. FILTER THEO THỜI GIAN
# =========================================================
valid_prediction_times = (
    history_df["PredictionTime"]
    .dropna()
)

if not valid_prediction_times.empty:
    minimum_date = (
        valid_prediction_times
        .min()
        .date()
    )

    maximum_date = (
        valid_prediction_times
        .max()
        .date()
    )

    selected_date_range = st.date_input(
        "Lọc theo khoảng thời gian",
        value=(
            minimum_date,
            maximum_date,
        ),
        min_value=minimum_date,
        max_value=maximum_date,
        format="DD/MM/YYYY",
    )

else:
    selected_date_range = (
        date.today(),
        date.today(),
    )


# =========================================================
# 6. ÁP DỤNG BỘ LỌC
# =========================================================
filtered_df = history_df.copy()


# Search chỉ theo mã khách hàng
if search_keyword.strip():
    keyword = search_keyword.strip()

    filtered_df = filtered_df[
        filtered_df["CustomerID"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )
    ]


# Lọc theo cụm
# if selected_clusters:
#     filtered_df = filtered_df[
#         filtered_df["Cluster"]
#         .isin(selected_clusters)
#     ]

# else:
#     filtered_df = filtered_df.iloc[0:0]
if selected_cluster_names:
    filtered_df = filtered_df[
        filtered_df["ClusterName"]
        .isin(selected_cluster_names)
    ]

else:
    filtered_df = filtered_df.iloc[0:0]


# Lọc theo loại Single hoặc Batch
if selected_prediction_types:
    filtered_df = filtered_df[
        filtered_df["PredictionType"]
        .isin(selected_prediction_types)
    ]

else:
    filtered_df = filtered_df.iloc[0:0]


# Lọc theo thời gian
if (
    isinstance(selected_date_range, tuple)
    and len(selected_date_range) == 2
):
    start_date, end_date = selected_date_range

    filtered_df = filtered_df[
        filtered_df["PredictionTime"]
        .dt.date
        .between(
            start_date,
            end_date,
        )
    ]


# =========================================================
# 7. KPI SAU KHI LỌC
# =========================================================
st.divider()

result_col_1, result_col_2, result_col_3 = st.columns(3)

result_col_1.metric(
    "Bản ghi hiển thị",
    f"{len(filtered_df):,}",
)

result_col_2.metric(
    "Khách hàng phù hợp",
    f"{filtered_df['CustomerID'].nunique():,}",
)

# result_col_3.metric(
#     "Cụm xuất hiện",
#     f"{filtered_df['Cluster'].nunique():,}",
# )
result_col_3.metric(
    "Phân khúc xuất hiện",
    f"{filtered_df['ClusterName'].nunique():,}",
)

# =========================================================
# 8. BẢNG LỊCH SỬ
# =========================================================
st.subheader("Danh sách lịch sử")


display_columns = [
    "PredictionTime",
    "PredictionType",
    "CustomerID",
    "LastPurchaseDate",
    "Recency",
    "Frequency",
    "Monetary",
    "Cluster",
    "ClusterName",
]


display_df = (
    filtered_df[
        display_columns
    ]
    .sort_values(
        "PredictionTime",
        ascending=False,
    )
    .copy()
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_order=display_columns,
    column_config={
        "PredictionTime": (
            st.column_config.DatetimeColumn(
                "Thời gian phân tích",
                format="DD/MM/YYYY HH:mm:ss",
            )
        ),
        "PredictionType": (
            st.column_config.TextColumn(
                "Loại phân tích",
            )
        ),
        "CustomerID": (
            st.column_config.TextColumn(
                "Mã khách hàng",
            )
        ),
        "LastPurchaseDate": (
            st.column_config.DateColumn(
                "Ngày mua gần nhất",
                format="DD/MM/YYYY",
            )
        ),
        "Recency": (
            st.column_config.NumberColumn(
                "Recency",
                format="%d ngày",
            )
        ),
        "Frequency": (
            st.column_config.NumberColumn(
                "Frequency",
                format="%d",
            )
        ),
        "Monetary": (
            st.column_config.NumberColumn(
                "Monetary",
                format="%.0f ₫",
            )
        ),
        "Cluster": (
            st.column_config.NumberColumn(
                "Cụm",
                format="%d",
            )
        ),
        "ClusterName": (
            st.column_config.TextColumn(
                "Tên phân khúc",
            )
        ),
        
    },
)


if filtered_df.empty:
    st.warning(
        "Không tìm thấy lịch sử phù hợp "
        "với điều kiện tìm kiếm và bộ lọc."
    )


# =========================================================
# 9. XUẤT CSV
# =========================================================
download_df = display_df.copy()

download_df["PredictionTime"] = (
    download_df["PredictionTime"]
    .dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)

download_df["LastPurchaseDate"] = (
    pd.to_datetime(
        download_df["LastPurchaseDate"],
        errors="coerce",
    )
    .dt.strftime("%Y-%m-%d")
)

csv_data = download_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Xuất lịch sử đã lọc (.CSV)",
    data=csv_data,
    file_name="prediction_history_filtered.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
    disabled=filtered_df.empty,
)


# =========================================================
# 10. XÓA LỊCH SỬ
# =========================================================
with st.expander(
    "Quản lý dữ liệu lịch sử",
    expanded=False,
):
    st.warning(
        "Thao tác này sẽ xóa toàn bộ lịch sử phân tích."
    )

    confirm_delete = st.checkbox(
        "Tôi xác nhận muốn xóa toàn bộ lịch sử"
    )

    if st.button(
        "Xóa toàn bộ lịch sử",
        disabled=not confirm_delete,
    ):
        clear_prediction_history()

        st.success(
            "Đã xóa toàn bộ lịch sử."
        )

        st.rerun()