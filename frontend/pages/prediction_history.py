from datetime import date

import pandas as pd
import streamlit as st

from utils.prediction_utils import (
    APIConnectionError,
    APIResponseError,
    is_backend_running,
    load_prediction_history,
)

# =========================================================
# KIỂM TRA BACKEND
# =========================================================
if not is_backend_running():
    st.error(
        "Không thể kết nối tới FastAPI Backend."
    )
    st.stop()
# =========================================================
# 1. TIÊU ĐỀ
# =========================================================
st.title("Lịch sử phân tích dữ liệu")

st.caption(
    "Tra cứu lịch sử phân nhóm khách hàng "
    "được lưu trong cơ sở dữ liệu MySQL."
)


# =========================================================
# 2. NÚT LÀM MỚI
# =========================================================
if st.button(
    "🔄 Làm mới dữ liệu",
):
    st.rerun()


# =========================================================
# 3. LẤY DỮ LIỆU TỪ API
# =========================================================
try:
    history_df = load_prediction_history()

except (
    APIConnectionError,
    APIResponseError,
) as error:
    st.error(
        f"Không thể tải lịch sử: {error}"
    )
    st.stop()

except Exception as error:
    st.error(
        "Đã xảy ra lỗi khi tải lịch sử: "
        f"{error}"
    )
    st.stop()


if history_df.empty:
    st.info(
        "Chưa có lịch sử phân tích trong cơ sở dữ liệu. "
        "Hãy thực hiện phân nhóm đơn lẻ hoặc hàng loạt trước."
    )
    st.stop()


# =========================================================
# 4. KPI TỔNG QUAN
# =========================================================
total_records = len(history_df)

total_segments = (
    history_df["cluster_label"]
    .dropna()
    .nunique()
)

average_monetary = (
    history_df["monetary"]
    .dropna()
    .mean()
)

kpi_1, kpi_2, kpi_3 = st.columns(3)

kpi_1.metric(
    "Tổng bản ghi",
    f"{total_records:,}",
)

kpi_2.metric(
    "Số phân khúc",
    f"{total_segments:,}",
)

kpi_3.metric(
    "Chi tiêu trung bình",
    (
        f"{average_monetary:,.0f} $"
        if pd.notna(average_monetary)
        else "0 ₫"
    ),
)


# =========================================================
# 5. BỘ LỌC
# =========================================================
st.subheader("Tìm kiếm và bộ lọc")

segment_options = sorted(
    history_df["cluster_label"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_segments = st.multiselect(
    "Lọc theo phân khúc",
    options=segment_options,
)


# =========================================================
# 6. LỌC NGÀY
# =========================================================
valid_times = (
    history_df["created_at"]
    .dropna()
)

if not valid_times.empty:
    minimum_date = valid_times.min().date()
    maximum_date = valid_times.max().date()

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
# 7. ÁP DỤNG BỘ LỌC
# =========================================================
filtered_df = history_df.copy()

# Lọc theo danh sách phân khúc được chọn từ thanh multiselect
if selected_segments:
    filtered_df = filtered_df[
        filtered_df["cluster_label"].isin(selected_segments)
    ]

if selected_segments:
    filtered_df = filtered_df[
        filtered_df["cluster_label"].isin(
            selected_segments
        )
    ]


if (
    isinstance(selected_date_range, tuple)
    and len(selected_date_range) == 2
):
    start_date, end_date = selected_date_range

    filtered_df = filtered_df[
        filtered_df["created_at"]
        .dt.date
        .between(
            start_date,
            end_date,
        )
    ]


# =========================================================
# 8. KPI SAU LỌC
# =========================================================
st.divider()

filtered_average = (
    filtered_df["monetary"]
    .dropna()
    .mean()
)

average_recency = (
    filtered_df["recency"]
    .dropna()
    .mean()
)


# =========================================================
# 9. BẢNG DỮ LIỆU
# =========================================================
st.subheader("Danh sách lịch sử")

display_columns = [
    "customer_id",
    "created_at",
    "recency",
    "frequency",
    "monetary",
    "cluster_id",
    "cluster_label",
]

if filtered_df.empty:
    st.warning(
        "Không tìm thấy lịch sử phù hợp "
        "với điều kiện lọc."
    )

    display_df = pd.DataFrame(
        columns=display_columns
    )

else:
    display_df = (
        filtered_df[display_columns]
        .sort_values(
            "created_at",
            ascending=False,
        )
        .copy()
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "customer_id": (
            st.column_config.TextColumn(
                "Mã khách hàng",
            )
        ),
        "created_at": (
            st.column_config.DatetimeColumn(
                "Thời gian phân tích",
                format="DD/MM/YYYY HH:mm:ss",
            )
        ),
        "recency": (
            st.column_config.NumberColumn(
                "Lần cuối mua",
                format="%.0f ngày",
            )
        ),
        "frequency": (
            st.column_config.NumberColumn(
                "Số lần mua",
                format="%.0f",
            )
        ),
        "monetary": (
            st.column_config.NumberColumn(
                "Tổng chi tiêu",
                format="%.0f $",
            )
        ),
        "cluster_id": (
            st.column_config.NumberColumn(
                "Mã cụm",
                format="%d",
            )
        ),
        "cluster_label": (
            st.column_config.TextColumn(
                "Tên phân khúc",
            )
        ),
    },
)


# =========================================================
# 10. XUẤT CSV
# =========================================================
download_df = display_df.copy()

if not download_df.empty:
    download_df["created_at"] = (
        download_df["created_at"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

csv_data = download_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "Xuất lịch sử đã lọc (.CSV)",
    data=csv_data,
    file_name=(
        f"prediction_history_{date.today()}.csv"
    ),
    mime="text/csv",
    type="primary",
    use_container_width=True,
    disabled=download_df.empty,
)
