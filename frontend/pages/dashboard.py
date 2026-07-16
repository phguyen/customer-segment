import pandas as pd
import streamlit as st

from utils.history_utils import load_prediction_history


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================
st.title("Tổng quan khách hàng")

st.caption(
    "Theo dõi các chỉ số và đặc điểm chính của từng "
    "phân khúc khách hàng."
)


# =========================================================
# 2. ĐỌC VÀ CHUẨN HÓA DỮ LIỆU
# =========================================================
try:
    history_df = load_prediction_history()

except Exception as error:
    st.error(f"Không thể tải dữ liệu lịch sử: {error}")
    st.stop()


if history_df is None:
    history_df = pd.DataFrame()


required_columns = [
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


for column in required_columns:
    if column not in history_df.columns:
        history_df[column] = None


# Chuyển kiểu dữ liệu
history_df["PredictionTime"] = pd.to_datetime(
    history_df["PredictionTime"],
    errors="coerce",
)

history_df["Recency"] = pd.to_numeric(
    history_df["Recency"],
    errors="coerce",
)

history_df["Frequency"] = pd.to_numeric(
    history_df["Frequency"],
    errors="coerce",
)

history_df["Monetary"] = pd.to_numeric(
    history_df["Monetary"],
    errors="coerce",
)

history_df["Cluster"] = pd.to_numeric(
    history_df["Cluster"],
    errors="coerce",
)

history_df["CustomerID"] = (
    history_df["CustomerID"]
    .fillna("")
    .astype(str)
)

history_df["ClusterName"] = (
    history_df["ClusterName"]
    .fillna("Chưa xác định")
    .astype(str)
)


# =========================================================
# 3. TRƯỜNG HỢP CHƯA CÓ DỮ LIỆU
# =========================================================
if history_df.empty:
    st.info(
        "Chưa có dữ liệu phân tích. Hãy thực hiện dự đoán tại "
        "trang Phân nhóm đơn lẻ hoặc Phân nhóm hàng loạt."
    )

    placeholder_1, placeholder_2, placeholder_3, placeholder_4 = (
        st.columns(4)
    )

    placeholder_1.metric("Lượt phân tích", "0")
    placeholder_2.metric("Khách hàng", "0")
    placeholder_3.metric("Phân khúc", "0")
    placeholder_4.metric("Tổng Monetary", "0 ₫")

    st.stop()


# =========================================================
# 4. BỘ LỌC
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

    available_types = sorted(
        history_df["PredictionType"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_clusters = filter_col_1.multiselect(
        "Phân khúc khách hàng",
        options=available_clusters,
        default=available_clusters,
    )

    selected_types = filter_col_2.multiselect(
        "Loại phân tích",
        options=available_types,
        default=available_types,
    )

    search_customer = filter_col_3.text_input(
        "Tìm mã khách hàng",
        placeholder="Ví dụ: CUST001",
    )


filtered_df = history_df.copy()


if selected_clusters:
    filtered_df = filtered_df[
        filtered_df["ClusterName"].isin(selected_clusters)
    ]


if selected_types:
    filtered_df = filtered_df[
        filtered_df["PredictionType"].isin(selected_types)
    ]


if search_customer.strip():
    filtered_df = filtered_df[
        filtered_df["CustomerID"].str.contains(
            search_customer.strip(),
            case=False,
            na=False,
        )
    ]


# =========================================================
# 5. KPI TỔNG QUAN
# =========================================================
total_predictions = len(filtered_df)

total_customers = filtered_df["CustomerID"].replace(
    "",
    pd.NA,
).nunique()

total_clusters = filtered_df["Cluster"].dropna().nunique()

total_monetary = filtered_df["Monetary"].fillna(0).sum()


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Lượt phân tích",
    f"{total_predictions:,}",
)

metric_2.metric(
    "Khách hàng",
    f"{total_customers:,}",
)

metric_3.metric(
    "Phân khúc",
    f"{total_clusters:,}",
)

metric_4.metric(
    "Tổng Monetary",
    f"{total_monetary:,.0f} ₫",
)


st.divider()


# =========================================================
# 6. BIỂU ĐỒ PHÂN BỐ PHÂN KHÚC
# =========================================================
chart_col_1, chart_col_2 = st.columns(2, gap="large")


with chart_col_1:
    with st.container(border=True):
        st.subheader("Phân bố khách hàng theo phân khúc")

        cluster_distribution = (
            filtered_df.groupby(
                "ClusterName",
                dropna=False,
            )
            .size()
            .reset_index(name="Số khách hàng")
            .sort_values(
                "Số khách hàng",
                ascending=False,
            )
        )

        if cluster_distribution.empty:
            st.info("Không có dữ liệu phù hợp với bộ lọc.")

        else:
            st.bar_chart(
                cluster_distribution.set_index(
                    "ClusterName"
                )["Số khách hàng"],
                use_container_width=True,
            )


with chart_col_2:
    with st.container(border=True):
        st.subheader("Giá trị Monetary theo phân khúc")

        monetary_by_cluster = (
            filtered_df.groupby(
                "ClusterName",
                dropna=False,
            )["Monetary"]
            .mean()
            .fillna(0)
            .sort_values(ascending=False)
        )

        if monetary_by_cluster.empty:
            st.info("Không có dữ liệu phù hợp với bộ lọc.")

        else:
            st.bar_chart(
                monetary_by_cluster,
                use_container_width=True,
            )


# =========================================================
# 7. RFM TRUNG BÌNH THEO PHÂN KHÚC
# =========================================================
with st.container(border=True):
    st.subheader("Chỉ số RFM trung bình theo phân khúc")

    rfm_summary = (
        filtered_df.groupby(
            [
                "Cluster",
                "ClusterName",
            ],
            dropna=False,
        )[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ]
        .mean()
        .reset_index()
    )

    if rfm_summary.empty:
        st.info("Không có dữ liệu RFM phù hợp.")

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
                    format="%d",
                ),
                "ClusterName": "Tên phân khúc",
                "Recency": st.column_config.NumberColumn(
                    "Recency trung bình",
                    format="%.1f ngày",
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency trung bình",
                    format="%.1f",
                ),
                "Monetary": st.column_config.NumberColumn(
                    "Monetary trung bình",
                    format="%.0f ₫",
                ),
            },
        )


# =========================================================
# 8. LỊCH SỬ PHÂN TÍCH GẦN NHẤT
# =========================================================
with st.container(border=True):
    st.subheader("Lịch sử phân tích gần nhất")

    recent_df = (
        filtered_df.sort_values(
            "PredictionTime",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    if recent_df.empty:
        st.info("Không có dữ liệu lịch sử phù hợp.")

    else:
        recent_df["PredictionTime"] = (
            recent_df["PredictionTime"]
            .dt.strftime("%d/%m/%Y %H:%M")
            .fillna("")
        )

        recent_df["Cluster"] = (
            recent_df["Cluster"]
            .fillna(-1)
            .astype(int)
        )

        recent_df["Recency"] = (
            recent_df["Recency"]
            .fillna(0)
            .astype(int)
        )

        recent_df["Frequency"] = (
            recent_df["Frequency"]
            .fillna(0)
            .astype(int)
        )

        recent_df["Monetary"] = (
            recent_df["Monetary"]
            .fillna(0)
            .astype(float)
        )

        display_columns = [
            "PredictionTime",
            "PredictionType",
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "Cluster",
            "ClusterName",
        ]

        st.dataframe(
            recent_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "PredictionTime": "Thời gian",
                "PredictionType": "Loại phân tích",
                "CustomerID": "Mã khách hàng",
                "Recency": st.column_config.NumberColumn(
                    "Recency",
                    format="%d ngày",
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency",
                    format="%d",
                ),
                "Monetary": st.column_config.NumberColumn(
                    "Monetary",
                    format="%.0f ₫",
                ),
                "Cluster": st.column_config.NumberColumn(
                    "Mã cụm",
                    format="%d",
                ),
                "ClusterName": "Tên phân khúc",
            },
        )