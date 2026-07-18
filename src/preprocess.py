import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


def data_loader(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy tập tin dữ liệu tại: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Ép kiểu dữ liệu ngày tháng, errors='coerce' để biến các chuỗi lỗi thành NaT
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    return df


def clean_data(df):
    print("\n=== BƯỚC 1: TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU ===")

    # 1. Loại bỏ các bản ghi trùng lặp hoàn toàn (Thực hiện trước khi drop cột sản phẩm)
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f" -> Loại bỏ dòng trùng lặp hoàn toàn: Đã xóa {before - df.shape[0]:,} dòng -> Còn {df.shape[0]:,} dòng")

    # 2. Loại bỏ dòng có Quantity <= 0
    before = df.shape[0]
    df = df[df['Quantity'] > 0]
    print(f" -> Loại bỏ Quantity <= 0: Đã xóa {before - df.shape[0]:,} dòng -> Còn {df.shape[0]:,} dòng")

    # 3. Loại bỏ dòng có UnitPrice <= 0
    before = df.shape[0]
    df = df[df['UnitPrice'] > 0]
    print(f" -> Loại bỏ UnitPrice <= 0: Đã xóa {before - df.shape[0]:,} dòng -> Còn {df.shape[0]:,} dòng")

    # 4. Loại bỏ dòng có CustomerID rỗng hoặc ngày tháng bị lỗi (NaT)
    before = df.shape[0]
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    print(f" -> Loại bỏ dòng khuyết CustomerID/InvoiceDate: Đã xóa {before - df.shape[0]:,} dòng -> Còn {df.shape[0]:,} dòng")

    # 5. Tính toán trường đặc trưng doanh thu: TotalPrice = Quantity * UnitPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df


def build_rfm_features(df, processed_dir,snapshot_date=None):
    print("\n=== BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG HÀNH VI KHÁCH HÀNG (RFM) ===")

    # Nếu k truyền gì thì thiết lập ngày mốc để tính Recency là Ngày giao dịch cuối cùng trong file
    # ĐỌC TỚI ĐÂY THÌ NHẮN T GIẢI THÍCH
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max()
    
    print(f"Ngày mốc tính Recency: {snapshot_date}")

    # Gom nhóm theo CustomerID để tính toán các chỉ số RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,   # Recency (Khoảng cách ngày)
        'InvoiceNo': 'nunique',                                     # Frequency (Số hóa đơn độc nhất)
        'TotalPrice': 'sum'                                         # Monetary (Tổng tiền chi tiêu trọn đời)
    }).reset_index()

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)

    print(f"Tổng số lượng khách hàng thu được: {rfm.shape[0]}")

    # Lưu bảng RFM đầy đủ. LƯU Ý: từ khi chuyển sang CLIP outlier (không xóa khách hàng nào
    # nữa), file này và customer_segmentation.csv sẽ có CÙNG số khách hàng -- không còn khách
    # nào bị coi là "outlier bị loại khỏi train" nữa, chỉ có giá trị R/F/M bị kẹp về biên hợp lý.
    full_rfm_path = os.path.join(processed_dir, 'customer_segmentation_full.csv')
    rfm.to_csv(full_rfm_path, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu tập dữ liệu RFM đầy đủ tại: {full_rfm_path}")

    return rfm


def handle_outliers_and_transform(rfm, processed_dir):
    print("\n=== BƯỚC 3: GIỚI HẠN NGOẠI LAI (CLIP) & BIẾN ĐỔI CHUẨN HÓA ===")

    # 1. Giới hạn (clip) outlier bằng phương pháp IQR - KHÔNG xóa khách hàng, chỉ kẹp giá trị
    # cực đoan về biên [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Nhờ vậy khách hàng chi tiêu/mua sắm cực
    # đoan vẫn được mô hình "nhìn thấy" lúc train, tâm cụm phản ánh sự tồn tại của họ thay vì
    # coi như không có, đồng thời không cần predict riêng cho nhóm "outlier bị loại" nữa.
    cols_to_clean = ['Recency', 'Frequency', 'Monetary']
    rfm_cleaned = rfm.copy()

    for col in cols_to_clean:
        Q1 = rfm_cleaned[col].quantile(0.25)
        Q3 = rfm_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        num_clipped = ((rfm_cleaned[col] < lower_bound) | (rfm_cleaned[col] > upper_bound)).sum()
        rfm_cleaned[col] = rfm_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        print(f" -> Cột '{col}': đã kẹp {num_clipped:,} giá trị cực đoan về khoảng "
              f"[{lower_bound:.2f}, {upper_bound:.2f}]")

    print(f" -> Tổng số khách hàng (giữ nguyên, không xóa ai): {rfm_cleaned.shape[0]:,}")

    # Lưu bảng RFM đơn vị gốc (đã clip) - Phục vụ phân tích đặc điểm phân cụm sau này
    raw_rfm_path = os.path.join(processed_dir, 'customer_segmentation.csv')
    rfm_cleaned.to_csv(raw_rfm_path, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu bảng RFM đã xử lý (đơn vị gốc) tại: {raw_rfm_path}")

    # 2. Biến đổi phân phối Log-transform (Khử lệch phải cho các thuật toán khoảng cách)
    rfm_log = rfm_cleaned[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)

    # 3. Chuẩn hóa phân phối Z-score đưa về Mean=0, Std=1 (StandardScaler)
    scaler = StandardScaler()
    rfm_scaled_array = scaler.fit_transform(rfm_log)

    rfm_scaled_df = pd.DataFrame(rfm_scaled_array, columns=['Recency', 'Frequency', 'Monetary'])
    rfm_scaled_df.insert(0, 'CustomerID', rfm_cleaned['CustomerID'].values)

    # Đóng gói và lưu trữ đối tượng Scaler pkl để tái sử dụng
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/rfm_scaler.pkl')
    print("-> Đã đóng gói và lưu bộ chuẩn hóa tại: models/rfm_scaler.pkl")

    # Lưu tập dữ liệu số đã xử lý hoàn chỉnh chuẩn bị đưa vào huấn luyện mô hình học máy
    scaled_data_path = os.path.join(processed_dir, 'customer_segmentation_scaled.csv')
    rfm_scaled_df.to_csv(scaled_data_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu tập dữ liệu số chuẩn hóa hoàn chỉnh tại: {scaled_data_path}")

    print("\n=== PIPELINE HOÀN THÀNH THÀNH CÔNG ===")


if __name__ == "__main__":
    # Cấu hình các đường dẫn thư mục lưu trữ đầu ra
    PROCESSED_DATA_DIR = "data/processed"
    RAW_DATA_PATH = "data/raw/Online Retail.csv"

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Thực thi tuần tự các bước xử lý trong Pipeline
    raw_df = data_loader(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    rfm_features = build_rfm_features(cleaned_df, PROCESSED_DATA_DIR)
    handle_outliers_and_transform(rfm_features, PROCESSED_DATA_DIR)