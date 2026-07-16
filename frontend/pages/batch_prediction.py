from datetime import datetime
import time

import pandas as pd
import streamlit as st
from utils.history_utils import (
    append_prediction_history,
)


# =========================================================
# 1. CÁC CỘT BẮT BUỘC
# =========================================================
REQUIRED_COLUMNS = {
    "CustomerID",
    "InvoiceNo",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
}


# =========================================================
# 2. HÀM ĐỌC FILE CSV
# =========================================================
def read_csv_file(uploaded_file):
    """
    Đọc file CSV.
    Thử UTF-8 trước, nếu lỗi mã hóa thì dùng latin-1.
    """

    try:
        return pd.read_csv(uploaded_file)

    except UnicodeDecodeError:
        uploaded_file.seek(0)

        return pd.read_csv(
            uploaded_file,
            encoding="latin-1",
        )


# =========================================================
# 3. HÀM LÀM SẠCH VÀ TẠO RFM
# =========================================================
def create_rfm_data(raw_df):
    """
    Tiền xử lý dữ liệu theo đúng pipeline của nhóm:

    1. Xóa dòng trùng.
    2. Xóa dòng thiếu CustomerID hoặc InvoiceDate.
    3. Chuyển InvoiceDate về datetime.
    4. Chuyển Quantity và UnitPrice về dạng số.
    5. Lọc Quantity > 0 và UnitPrice > 0.
    6. Tính Revenue.
    7. Tính Recency, Frequency, Monetary.
    """

    missing_columns = REQUIRED_COLUMNS - set(raw_df.columns)

    if missing_columns:
        raise ValueError(
            "File đang thiếu các cột bắt buộc: "
            + ", ".join(sorted(missing_columns))
        )

    df_cleaned = raw_df.copy()

    original_rows = len(df_cleaned)

    # 1. Xóa dữ liệu trùng
    df_cleaned = df_cleaned.drop_duplicates()

    duplicate_rows_removed = (
        original_rows - len(df_cleaned)
    )

    # 2. Xóa dòng thiếu dữ liệu quan trọng
    rows_before_missing = len(df_cleaned)

    df_cleaned = df_cleaned.dropna(
        subset=[
            "CustomerID",
            "InvoiceDate",
        ]
    )

    missing_rows_removed = (
        rows_before_missing - len(df_cleaned)
    )

    # 3. Chuyển InvoiceDate sang datetime
    df_cleaned["InvoiceDate"] = pd.to_datetime(
        df_cleaned["InvoiceDate"],
        errors="coerce",
    )

    rows_before_invalid_date = len(df_cleaned)

    df_cleaned = df_cleaned.dropna(
        subset=["InvoiceDate"]
    )

    invalid_date_rows_removed = (
        rows_before_invalid_date - len(df_cleaned)
    )

    # 4. Chuyển Quantity và UnitPrice sang số
    df_cleaned["Quantity"] = pd.to_numeric(
        df_cleaned["Quantity"],
        errors="coerce",
    )

    df_cleaned["UnitPrice"] = pd.to_numeric(
        df_cleaned["UnitPrice"],
        errors="coerce",
    )

    rows_before_invalid_number = len(df_cleaned)

    df_cleaned = df_cleaned.dropna(
        subset=[
            "Quantity",
            "UnitPrice",
        ]
    )

    invalid_number_rows_removed = (
        rows_before_invalid_number - len(df_cleaned)
    )

    # 5. Lọc giao dịch không hợp lệ
    rows_before_invalid_transaction = len(df_cleaned)

    df_cleaned = df_cleaned[
        (df_cleaned["Quantity"] > 0)
        & (df_cleaned["UnitPrice"] > 0)
    ]

    invalid_transaction_rows_removed = (
        rows_before_invalid_transaction
        - len(df_cleaned)
    )

    if df_cleaned.empty:
        raise ValueError(
            "Không còn giao dịch hợp lệ sau khi làm sạch dữ liệu."
        )

    # 6. Tính Revenue
    df_cleaned["Revenue"] = (
        df_cleaned["Quantity"]
        * df_cleaned["UnitPrice"]
    )

    # 7. Dùng max_date giống notebook của nhóm
    max_date = df_cleaned["InvoiceDate"].max()

    customer_rfm = (
        df_cleaned.groupby("CustomerID")
        .agg(
            LastPurchaseDate=(
                "InvoiceDate",
                "max",
            ),
            Recency=(
                "InvoiceDate",
                lambda values: (
                    max_date - values.max()
                ).days,
            ),
            Frequency=(
                "InvoiceNo",
                "nunique",
            ),
            Monetary=(
                "Revenue",
                "sum",
            ),
        )
        .reset_index()
    )

    cleaning_report = {
        "original_rows": original_rows,
        "cleaned_rows": len(df_cleaned),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_rows_removed": missing_rows_removed,
        "invalid_date_rows_removed": invalid_date_rows_removed,
        "invalid_number_rows_removed": invalid_number_rows_removed,
        "invalid_transaction_rows_removed": (
            invalid_transaction_rows_removed
        ),
        "reference_date": max_date,
    }

    return customer_rfm, cleaning_report


# =========================================================
# 4. HÀM PHÂN CỤM DEMO
# =========================================================
def predict_customer(recency, frequency, monetary):
    """
    Logic demo.

    Sau này thay phần này bằng:
    scaler.transform()
    model.predict()
    """

    if (
        recency <= 30
        and frequency >= 8
        and monetary >= 5_000_000
    ):
        return 0, "Khách hàng giá trị cao"

    if (
        recency <= 90
        and frequency >= 4
        and monetary >= 1_500_000
    ):
        return 1, "Khách hàng tiềm năng"

    if recency > 180 or frequency <= 1:
        return 3, "Khách hàng có nguy cơ rời bỏ"

    return 2, "Khách hàng cần chăm sóc"


# =========================================================
# 5. TIÊU ĐỀ TRANG
# =========================================================
st.title("Phân nhóm khách hàng hàng loạt")

st.caption(
    "Tải lên lịch sử giao dịch thô để làm sạch dữ liệu, "
    "tạo bảng RFM và phân nhóm nhiều khách hàng cùng lúc."
)

# =========================================================
# 6. HƯỚNG DẪN FILE
# =========================================================
with st.expander(
    "Xem yêu cầu định dạng file CSV",
    expanded=False,
):
    st.write(
        "File lịch sử giao dịch cần có các cột sau:"
    )

    required_columns_df = pd.DataFrame(
        {
            "Tên cột": [
                "CustomerID",
                "InvoiceNo",
                "InvoiceDate",
                "Quantity",
                "UnitPrice",
            ],
            "Ý nghĩa": [
                "Mã khách hàng",
                "Mã hóa đơn",
                "Ngày giao dịch",
                "Số lượng sản phẩm",
                "Đơn giá sản phẩm",
            ],
            "Ví dụ": [
                "CUST001",
                "INV001",
                "2026-07-01",
                "2",
                "250000",
            ],
        }
    )

    st.dataframe(
        required_columns_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 7. KHU VỰC UPLOAD
# =========================================================
with st.container(border=True):
    st.subheader("Tải dữ liệu giao dịch")

    st.caption(
        "Kéo thả hoặc chọn file CSV chứa lịch sử giao dịch."
    )

    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=["csv"],
        help="File phải chứa dữ liệu giao dịch thô.",
    )


# =========================================================
# 8. ĐỌC VÀ KIỂM TRA FILE
# =========================================================
if uploaded_file is not None:
    try:
        raw_df = read_csv_file(uploaded_file)

        st.success(
            f"Đã tải file thành công: {uploaded_file.name}"
        )

        file_col_1, file_col_2, file_col_3 = st.columns(3)

        file_col_1.metric(
            "Số dòng",
            f"{len(raw_df):,}",
        )

        file_col_2.metric(
            "Số cột",
            len(raw_df.columns),
        )

        file_col_3.metric(
            "Dung lượng",
            f"{uploaded_file.size / 1024:.1f} KB",
        )

        # =============================================
        # PREVIEW FILE
        # =============================================
        with st.container(border=True):
            st.subheader("Xem trước dữ liệu")

            st.caption(
                "Hiển thị 20 dòng đầu tiên trong file."
            )

            st.dataframe(
                raw_df.head(20),
                use_container_width=True,
                hide_index=True,
            )

        # =============================================
        # KIỂM TRA CỘT
        # =============================================
        missing_columns = (
            REQUIRED_COLUMNS
            - set(raw_df.columns)
        )

        if missing_columns:
            st.error(
                "File chưa đúng cấu trúc. Thiếu các cột: "
                + ", ".join(sorted(missing_columns))
            )

        else:
            st.success(
                "Cấu trúc file hợp lệ và sẵn sàng phân tích."
            )

            analyze_button = st.button(
                "Bắt đầu phân tích",
                type="primary",
                use_container_width=True,
            )

            if analyze_button:
                progress_bar = st.progress(
                    0,
                    text="Đang kiểm tra dữ liệu...",
                )

                try:
                    time.sleep(0.2)

                    progress_bar.progress(
                        20,
                        text="Đang loại bỏ dữ liệu trùng...",
                    )

                    time.sleep(0.2)

                    progress_bar.progress(
                        40,
                        text="Đang xử lý dữ liệu thiếu và sai định dạng...",
                    )

                    rfm_df, cleaning_report = create_rfm_data(
                        raw_df
                    )

                    time.sleep(0.2)

                    progress_bar.progress(
                        60,
                        text="Đang tính Recency, Frequency và Monetary...",
                    )

                    clusters = []
                    cluster_names = []

                    for _, row in rfm_df.iterrows():
                        cluster, cluster_name = predict_customer(
                            recency=int(row["Recency"]),
                            frequency=int(row["Frequency"]),
                            monetary=float(row["Monetary"]),
                        )

                        clusters.append(cluster)
                        cluster_names.append(cluster_name)

                    time.sleep(0.2)

                    progress_bar.progress(
                        85,
                        text="Đang phân nhóm khách hàng...",
                    )

                    result_df = rfm_df.copy()
                    
                    result_df["Cluster"] = clusters
                    result_df["ClusterName"] = cluster_names

                    result_df["PredictionTime"] = (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )
                    result_df["PredictionType"] = "Batch"

                    append_prediction_history(
                        result_df
                    )

                    progress_bar.progress(
                        100,
                        text="Hoàn tất phân tích.",
                    )

                    st.session_state[
                        "batch_result"
                    ] = result_df

                    st.session_state[
                        "cleaning_report"
                    ] = cleaning_report

                    st.success(
                        f"Đã phân tích thành công "
                        f"{len(result_df):,} khách hàng."
                    )

                except Exception as error:
                    progress_bar.empty()

                    st.error(
                        f"Không thể xử lý dữ liệu: {error}"
                    )

    except Exception as error:
        st.error(
            f"Không thể đọc file CSV: {error}"
        )


# =========================================================
# 9. BÁO CÁO LÀM SẠCH DỮ LIỆU
# =========================================================
if "cleaning_report" in st.session_state:
    report = st.session_state["cleaning_report"]

    st.divider()

    st.subheader("Báo cáo làm sạch dữ liệu")

    report_col_1, report_col_2, report_col_3 = st.columns(3)

    report_col_1.metric(
        "Dữ liệu ban đầu",
        f"{report['original_rows']:,} dòng",
    )

    report_col_2.metric(
        "Dữ liệu hợp lệ",
        f"{report['cleaned_rows']:,} dòng",
    )

    total_removed = (
        report["original_rows"]
        - report["cleaned_rows"]
    )

    report_col_3.metric(
        "Đã loại bỏ",
        f"{total_removed:,} dòng",
    )

    with st.expander(
        "Xem chi tiết các dòng bị loại bỏ",
        expanded=False,
    ):
        detail_df = pd.DataFrame(
            {
                "Nguyên nhân": [
                    "Dữ liệu trùng",
                    "Thiếu CustomerID hoặc InvoiceDate",
                    "InvoiceDate không hợp lệ",
                    "Quantity hoặc UnitPrice không phải số",
                    "Quantity hoặc UnitPrice không hợp lệ",
                ],
                "Số dòng bị loại": [
                    report["duplicate_rows_removed"],
                    report["missing_rows_removed"],
                    report["invalid_date_rows_removed"],
                    report["invalid_number_rows_removed"],
                    report[
                        "invalid_transaction_rows_removed"
                    ],
                ],
            }
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            "**Ngày tham chiếu dùng tính Recency:**",
            report["reference_date"].strftime(
                "%d/%m/%Y"
            ),
        )


# =========================================================
# 10. HIỂN THỊ KẾT QUẢ
# =========================================================
if "batch_result" in st.session_state:
    result_df = st.session_state["batch_result"]

    st.divider()

    st.subheader("Kết quả phân nhóm")

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)

    summary_col_1.metric(
        "Tổng khách hàng",
        f"{result_df['CustomerID'].nunique():,}",
    )

    summary_col_2.metric(
        "Số phân khúc",
        result_df["Cluster"].nunique(),
    )

    summary_col_3.metric(
        "Tổng giá trị",
        (
            f"{result_df['Monetary'].sum() / 1_000_000:,.1f} "
            "triệu ₫"
        ),
    )

    result_tab, summary_tab = st.tabs(
        [
            "Danh sách khách hàng",
            "Tóm tắt theo phân khúc",
        ]
    )

    # =============================================
    # TAB 1: DANH SÁCH KHÁCH HÀNG
    # =============================================
    with result_tab:
        display_df = result_df.copy()

        display_df["LastPurchaseDate"] = (
            pd.to_datetime(
                display_df["LastPurchaseDate"]
            ).dt.strftime("%d/%m/%Y")
        )

        st.dataframe(
            display_df[
                [
                    "CustomerID",
                    "LastPurchaseDate",
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "Cluster",
                    "ClusterName",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "CustomerID": "Mã khách hàng",
                "LastPurchaseDate": "Ngày mua gần nhất",
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

    # =============================================
    # TAB 2: TÓM TẮT THEO CỤM
    # =============================================
    with summary_tab:
        cluster_summary = (
            result_df.groupby(
                [
                    "Cluster",
                    "ClusterName",
                ]
            )
            .agg(
                CustomerCount=(
                    "CustomerID",
                    "nunique",
                ),
                AverageRecency=(
                    "Recency",
                    "mean",
                ),
                AverageFrequency=(
                    "Frequency",
                    "mean",
                ),
                AverageMonetary=(
                    "Monetary",
                    "mean",
                ),
            )
            .round(2)
            .reset_index()
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cluster": "Mã cụm",
                "ClusterName": "Tên phân khúc",
                "CustomerCount": "Số khách hàng",
                "AverageRecency": (
                    "Recency trung bình"
                ),
                "AverageFrequency": (
                    "Frequency trung bình"
                ),
                "AverageMonetary": (
                    st.column_config.NumberColumn(
                        "Monetary trung bình",
                        format="%.0f ₫",
                    )
                ),
            },
        )

    # =============================================
    # DOWNLOAD FILE
    # =============================================
    download_df = result_df.copy()

    download_df["LastPurchaseDate"] = (
        pd.to_datetime(
            download_df["LastPurchaseDate"]
        ).dt.strftime("%Y-%m-%d")
    )

    csv_data = download_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Tải file kết quả (.CSV)",
        data=csv_data,
        file_name="customer_segmentation_result.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Kết quả hiện được tạo bằng logic phân cụm minh họa. "
        "Khi model hoàn thành, phần dự đoán sẽ được thay bằng "
        "scaler và mô hình K-Means thật."
    )