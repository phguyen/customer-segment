# customer-segment

## Khởi tạo & kích hoạt môi trường ảo

1. Truy cập folder customer-segment

   `cd  customer-segment`

2. Tạo virtual env

   `python -m venv venv`

   `venv\Scripts\activate`   # Windows

   `source venv/bin/activate`  # Mac/Linux

3. Cài thư viện

   `pip install -r requirements.txt`

## Cấu trúc dự án
customer-segment/
├── backend/                # Tầng API & Xử lý logic nghiệp vụ
│   ├── database/           # Quản lý kết nối & truy vấn CSDL PostgreSQL
│   ├── main.py             # Entry point, khởi tạo ứng dụng FastAPI
│   └── router.py           # Định nghĩa các endpoint (Health, Model-info, Predict, History)
├── frontend/               # Tầng giao diện người dùng (Streamlit Dashboard)
│   └── app.py              # Giao diện chính, gọi API và hiển thị kết quả
├── src/                    # Mã nguồn huấn luyện & tiền xử lý (Core)
│   ├── preprocess.py       # Xử lý làm sạch dữ liệu (ETL Pipeline)
│   └── train.py            # Huấn luyện mô hình & đóng gói model
|── notebooks/              # Khám phá DL & Thử nghiệm mô hình
├── models/                 # Lưu trữ artifacts (.pkl, .json)
├── data/                   # Quản lý tập dữ liệu (DL thô/ DL đã tiền xử lý)
└── requirements.txt        # Danh sách thư viện cần cài
