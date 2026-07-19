import os
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests


# =========================================================
# 1. CẤU HÌNH API
# =========================================================
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 30
BATCH_REQUEST_TIMEOUT = 180


# =========================================================
# 2. LỖI API
# =========================================================
class APIConnectionError(Exception):
    """Không thể kết nối tới FastAPI backend."""


class APIResponseError(Exception):
    """Backend trả về lỗi hoặc dữ liệu không hợp lệ."""


# =========================================================
# 3. XỬ LÝ PHẢN HỒI API
# =========================================================
def _extract_error_message(response: requests.Response) -> str:
    try:
        response_data = response.json()

        if isinstance(response_data, dict):
            detail = response_data.get("detail")

            if isinstance(detail, list):
                return "; ".join(
                    str(item.get("msg", item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in detail
                )

            if detail:
                return str(detail)

            message = response_data.get("message")

            if message:
                return str(message)

        return str(response_data)

    except ValueError:
        return response.text.strip() or (
            f"Backend trả về mã lỗi {response.status_code}."
        )


def _request(
    method: str,
    endpoint: str,
    *,
    timeout: int = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> dict:
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs,
        )

    except requests.exceptions.ConnectionError as error:
        raise APIConnectionError(
            "Không thể kết nối tới backend. "
            "Hãy kiểm tra FastAPI đã được chạy tại "
            f"{API_BASE_URL} hay chưa."
        ) from error

    except requests.exceptions.Timeout as error:
        raise APIConnectionError(
            "Backend phản hồi quá chậm hoặc yêu cầu đã hết thời gian chờ."
        ) from error

    except requests.exceptions.RequestException as error:
        raise APIConnectionError(
            f"Không thể gửi yêu cầu tới backend: {error}"
        ) from error

    if not response.ok:
        raise APIResponseError(
            _extract_error_message(response)
        )

    try:
        response_data = response.json()

    except ValueError as error:
        raise APIResponseError(
            "Backend trả về dữ liệu không đúng định dạng JSON."
        ) from error

    if not isinstance(response_data, dict):
        raise APIResponseError(
            "Dữ liệu backend trả về không đúng cấu trúc."
        )

    return response_data


# =========================================================
# 4. KIỂM TRA BACKEND
# =========================================================
def check_backend_health() -> dict:
    return _request(
        "GET",
        "/health",
        timeout=10,
    )


# =========================================================
# 5. LẤY THÔNG TIN MODEL
# =========================================================
def get_model_info() -> dict:
    return _request(
        "GET",
        "/model-info",
        timeout=10,
    )


# =========================================================
# 6. DỰ ĐOÁN ĐƠN LẺ
# =========================================================
def predict_customer(
    customer_id: str,
    last_purchase_date: date | datetime | str,
    frequency: float,
    monetary: float,
) -> dict:
    if isinstance(last_purchase_date, datetime):
        date_string = last_purchase_date.date().isoformat()

    elif isinstance(last_purchase_date, date):
        date_string = last_purchase_date.isoformat()

    else:
        date_string = str(last_purchase_date)

    payload = {
        "customer_id": customer_id,
        "last_purchase_date": date_string,
        "frequency": float(frequency),
        "monetary": float(monetary),
    }

    return _request(
        "POST",
        "/predict",
        json=payload,
    )


# =========================================================
# 7. DỰ ĐOÁN HÀNG LOẠT
# =========================================================
def predict_customer_file(
    uploaded_file,
) -> dict:
    uploaded_file.seek(0)

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    return _request(
        "POST",
        "/predict-file",
        files=files,
        timeout=BATCH_REQUEST_TIMEOUT,
    )


# =========================================================
# 8. LẤY LỊCH SỬ TỪ MYSQL
# =========================================================
def load_prediction_history() -> pd.DataFrame:
    response_data = _request(
        "GET",
        "/history",
    )

    history_rows = response_data.get(
        "history",
        [],
    )

    required_columns = [
        "id",
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "cluster_id",
        "cluster_label",
        "created_at",
    ]

    if not history_rows:
        return pd.DataFrame(
            columns=required_columns
        )

    history_df = pd.DataFrame(history_rows)

    for column in required_columns:
        if column not in history_df.columns:
            history_df[column] = None

    history_df["id"] = pd.to_numeric(
        history_df["id"],
        errors="coerce",
    )

    history_df["recency"] = pd.to_numeric(
        history_df["recency"],
        errors="coerce",
    )

    history_df["frequency"] = pd.to_numeric(
        history_df["frequency"],
        errors="coerce",
    )

    history_df["monetary"] = pd.to_numeric(
        history_df["monetary"],
        errors="coerce",
    )

    history_df["cluster_id"] = pd.to_numeric(
        history_df["cluster_id"],
        errors="coerce",
    )

    history_df["cluster_label"] = (
        history_df["cluster_label"]
        .fillna("Chưa xác định")
        .astype(str)
    )

    history_df["created_at"] = pd.to_datetime(
        history_df["created_at"],
        errors="coerce",
    )

    return history_df[required_columns]
# =========================================================
# 9. CÁC HÀM TƯƠNG THÍCH VỚI FRONTEND
# =========================================================
def is_backend_running() -> bool:
    """
    Kiểm tra FastAPI backend có đang hoạt động hay không.
    """

    try:
        health_data = check_backend_health()

        return health_data.get("status") == "healthy"

    except (
        APIConnectionError,
        APIResponseError,
    ):
        return False


def get_model_information() -> dict:
    """
    Tên hàm tương thích với trang model_information.py.
    """

    try:
        data = get_model_info()

        return {
            "success": True,
            "data": data,
            "error": None,
        }

    except (
        APIConnectionError,
        APIResponseError,
    ) as error:
        return {
            "success": False,
            "data": {},
            "error": str(error),
        }


def get_prediction_history() -> dict:
    """
    Lấy lịch sử dự đoán và trả về cấu trúc dùng cho dashboard.
    """

    try:
        history_df = load_prediction_history()

        history_rows = history_df.to_dict(
            orient="records"
        )

        # Chuyển Timestamp thành chuỗi để Streamlit có thể
        # hiển thị bằng st.json nếu cần.
        for row in history_rows:
            created_at = row.get("created_at")

            if pd.notna(created_at):
                row["created_at"] = (
                    created_at.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            else:
                row["created_at"] = None

        return {
            "success": True,
            "data": history_rows,
            "error": None,
        }

    except (
        APIConnectionError,
        APIResponseError,
    ) as error:
        return {
            "success": False,
            "data": [],
            "error": str(error),
        }