import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# =====================================================
# Cấu hình tải biến môi trường
# =====================================================
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# =====================================================
# Khởi tạo SQLAlchemy
# =====================================================
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
    bind=engine
)

Base = declarative_base()

# =====================================================
# Định nghĩa bảng
# =====================================================
class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), nullable=False)
    recency = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False)
    monetary = Column(Float, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    cluster_label = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now) # Sử dụng datetime.now để tránh cảnh báo utcnow

# Tạo bảng nếu chưa tồn tại
Base.metadata.create_all(bind=engine)

# =====================================================
# Lưu lịch sử dự đoán
# =====================================================
def log_to_db(customer_id: str, recency: float, frequency: float, monetary: float, cluster_id: int, cluster_label: str):
    session = SessionLocal()
    try:
        row = PredictionHistory(
            customer_id=str(customer_id),  # Ép kiểu sang chuỗi để an toàn cho cả hai luồng (đơn lẻ/file)
            recency=recency,
            frequency=frequency,
            monetary=monetary,
            cluster_id=cluster_id,
            cluster_label=cluster_label
        )
        session.add(row)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
# =====================================================
# Lấy lịch sử dự đoán
# =====================================================
def get_prediction_history_from_db():
    session = SessionLocal()
    try:
        rows = (
            session.query(PredictionHistory)
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )
        
        return [
            {
                "id": row.id,
                "customer_id": row.customer_id,
                "recency": row.recency,
                "frequency": row.frequency,
                "monetary": row.monetary,
                "cluster_id": row.cluster_id,
                "cluster_label": row.cluster_label,
                "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None
            }
            for row in rows
        ]
    finally:
        session.close()
