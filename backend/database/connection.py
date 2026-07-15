from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# =====================================================
# Cấu hình MySQL
# =====================================================

DB_USER = "root"
DB_PASSWORD = "123456"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "customer_segmentation"

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

    id = Column(Integer, primary_key=True, index=True)

    recency = Column(Float, nullable=False)

    frequency = Column(Float, nullable=False)

    monetary = Column(Float, nullable=False)

    cluster_id = Column(Integer, nullable=False)

    cluster_label = Column(String(100), nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# =====================================================
# Tạo bảng
# =====================================================

Base.metadata.create_all(bind=engine)

# =====================================================
# Lưu lịch sử dự đoán
# =====================================================

def log_to_db(
    recency: float,
    frequency: float,
    monetary: float,
    cluster_id: int,
    cluster_label: str
):
    session = SessionLocal()

    try:
        row = PredictionHistory(
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

        history = []

        for row in rows:

            history.append({

                "id": row.id,

                "recency": row.recency,

                "frequency": row.frequency,

                "monetary": row.monetary,

                "cluster_id": row.cluster_id,

                "cluster_label": row.cluster_label,

                "created_at": row.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            })

        return history

    finally:
        session.close()