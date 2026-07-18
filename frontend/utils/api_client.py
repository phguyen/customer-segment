import os

import requests

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


class APIError(Exception):
    pass


def api_get(endpoint: str):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=30,
        )

    except requests.exceptions.ConnectionError:
        raise APIError(
            "Không kết nối được Backend FastAPI."
        )

    except requests.exceptions.Timeout:
        raise APIError(
            "Backend phản hồi quá chậm."
        )

    if response.status_code != 200:
        try:
            detail = response.json()["detail"]
        except Exception:
            detail = response.text

        raise APIError(detail)

    return response.json()


def api_post(endpoint: str, json=None, files=None):
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=json,
            files=files,
            timeout=180,
        )

    except requests.exceptions.ConnectionError:
        raise APIError(
            "Không kết nối được Backend FastAPI."
        )

    except requests.exceptions.Timeout:
        raise APIError(
            "Backend phản hồi quá chậm."
        )

    if response.status_code != 200:
        try:
            detail = response.json()["detail"]
        except Exception:
            detail = response.text

        raise APIError(detail)

    return response.json()
# =========================================================
# CÁC HÀM API DÙNG CHO GIAO DIỆN STREAMLIT
# =========================================================
def get_model_information():
    """
    Lấy thông tin mô hình từ endpoint /model-info.

    Trả về cấu trúc thống nhất cho frontend:
    {
        "success": True/False,
        "data": {...},
        "error": None hoặc thông báo lỗi
    }
    """

    try:
        data = api_get("/model-info")

        return {
            "success": True,
            "data": data,
            "error": None,
        }

    except APIError as error:
        return {
            "success": False,
            "data": {},
            "error": str(error),
        }


def get_prediction_history():
    """
    Lấy lịch sử dự đoán từ endpoint /history.
    """

    try:
        response_data = api_get("/history")

        return {
            "success": True,
            "data": response_data.get("history", []),
            "error": None,
        }

    except APIError as error:
        return {
            "success": False,
            "data": [],
            "error": str(error),
        }


def is_backend_running():
    """
    Kiểm tra backend FastAPI có hoạt động hay không.
    """

    try:
        health_data = api_get("/health")

        return health_data.get("status") == "healthy"

    except APIError:
        return False