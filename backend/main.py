from fastapi import FastAPI
# Import router từ file router.py cùng cấp trong thư mục backend
from backend.router import router 

# 1. Khởi tạo ứng dụng FastAPI chính (đây chính là biến "app" mà uvicorn tìm kiếm)
app = FastAPI(
    title="Customer Segmentation API",
    description="REST API phục vụ mô hình Phân khúc khách hàng RFM - K-Means",
    version="1.0"
)

# 2. Nhúng toàn bộ các endpoint (routes) từ file router.py vào hệ thống
app.include_router(router)