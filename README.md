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
customer-segment/ <br>
├── backend/                # Tầng API & Xử lý logic nghiệp vụ <br>
│   ├── database/           # Quản lý kết nối & truy vấn CSDL MySQL <br>
│   ├── main.py             # Entry point, khởi tạo ứng dụng FastAPI<br>
│   └── router.py           # Định nghĩa các endpoint (Health, Model-info, Predict, History) <br>
├── frontend/               # Tầng giao diện người dùng (Streamlit Dashboard) <br>
│   └── app.py              # Giao diện chính, gọi API và hiển thị kết quả <br>
├── src/                    # Mã nguồn huấn luyện & tiền xử lý (Core) <br>
│   ├── preprocess.py       # Xử lý làm sạch dữ liệu (ETL Pipeline) <br>
│   └── train.py            # Huấn luyện mô hình & đóng gói model <br>
|── notebooks/              # Khám phá DL & Thử nghiệm mô hình <br>
├── models/                 # Lưu trữ model (.pkl, .json) <br>
├── data/                   # Quản lý tập dữ liệu (DL thô/ DL đã tiền xử lý) <br>
└── requirements.txt        # Danh sách thư viện cần cài <br>
