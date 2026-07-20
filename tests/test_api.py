from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"
TIMEOUT = 30


def test_health():
    """Kiểm tra backend và mô hình đã sẵn sàng."""

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_info():
    """Kiểm tra API trả thông tin mô hình K-Means."""

    response = requests.get(
        f"{BASE_URL}/model-info",
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["Algorithm"] == "K-Means"
    assert isinstance(data["Số cụm"], int)
    assert data["Số cụm"] > 0


def test_predict_valid_customer():
    """Kiểm tra dự đoán một khách hàng hợp lệ."""

    payload = {
        "customer_id": "TEST001",
        "last_purchase_date": "2024-01-01",
        "frequency": 5,
        "monetary": 2500,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["cluster_id"], int)
    assert isinstance(data["persona"], str)
    assert isinstance(data["description"], str)
    assert data["confidence"] in ["high", "low"]


def test_predict_missing_required_field():
    """Thiếu customer_id phải bị FastAPI từ chối."""

    payload = {
        "last_purchase_date": "2024-01-01",
        "frequency": 5,
        "monetary": 2500,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=TIMEOUT,
    )

    assert response.status_code == 422


def test_predict_invalid_date():
    """Ngày sai định dạng phải trả lỗi 400."""

    payload = {
        "customer_id": "TEST002",
        "last_purchase_date": "01-01-2024",
        "frequency": 5,
        "monetary": 2500,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=TIMEOUT,
    )

    assert response.status_code == 400


def test_predict_file_rejects_non_csv():
    """API không được chấp nhận file không phải CSV."""

    files = {
        "file": (
            "invalid.txt",
            b"this is not a csv file",
            "text/plain",
        )
    }

    response = requests.post(
        f"{BASE_URL}/predict-file",
        files=files,
        timeout=TIMEOUT,
    )

    assert response.status_code == 400


def test_predict_file_valid_csv():
    """Kiểm tra dự đoán hàng loạt bằng file CSV mẫu."""

    csv_path = Path(
        "frontend/data/"
        "sample_transactions_numeric_customerid.csv"
    )

    assert csv_path.exists(), (
        f"Không tìm thấy file test: {csv_path}"
    )

    with csv_path.open("rb") as csv_file:
        files = {
            "file": (
                csv_path.name,
                csv_file,
                "text/csv",
            )
        }

        response = requests.post(
            f"{BASE_URL}/predict-file",
            files=files,
            timeout=60,
        )

    assert response.status_code == 200

    data = response.json()

    assert "total_customers_segmented" in data
    assert "results" in data
    assert isinstance(data["results"], list)
    assert data["total_customers_segmented"] > 0

    first_result = data["results"][0]

    assert "customer_id" in first_result
    assert "cluster_id" in first_result
    assert "persona" in first_result
    assert "confidence" in first_result


def test_prediction_history():
    """Kiểm tra lấy lịch sử dự đoán từ MySQL."""

    response = requests.get(
        f"{BASE_URL}/history",
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert "history" in data
    assert isinstance(data["history"], list)
def test_predict_future_date():
    payload = {
        "customer_id": "TEST003",
        "last_purchase_date": "2099-01-01",
        "frequency": 5,
        "monetary": 2500,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=TIMEOUT,
    )

    assert response.status_code == 400
def test_predict_file_missing_columns():
    csv_content = b"CustomerID,Quantity\nC001,2\n"

    files = {
        "file": (
            "missing_columns.csv",
            csv_content,
            "text/csv",
        )
    }

    response = requests.post(
        f"{BASE_URL}/predict-file",
        files=files,
        timeout=TIMEOUT,
    )

    assert response.status_code == 400
    