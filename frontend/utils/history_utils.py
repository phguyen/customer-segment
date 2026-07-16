from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "data" / "prediction_history.csv"

HISTORY_COLUMNS = [
    "PredictionTime",
    "PredictionType",
    "CustomerID",
    "LastPurchaseDate",
    "Recency",
    "Frequency",
    "Monetary",
    "Cluster",
    "ClusterName",
]


def initialize_history_file():
    """
    Tạo file lịch sử đúng cấu trúc nếu chưa tồn tại
    hoặc file đang rỗng.
    """
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not HISTORY_PATH.exists()
        or HISTORY_PATH.stat().st_size == 0
    ):
        pd.DataFrame(
            columns=HISTORY_COLUMNS
        ).to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )


def ensure_file_ends_with_newline():
    """
    Bảo đảm dữ liệu mới không bị nối vào cuối dòng tiêu đề.
    """
    if not HISTORY_PATH.exists():
        return

    if HISTORY_PATH.stat().st_size == 0:
        return

    with HISTORY_PATH.open("rb") as file:
        file.seek(-1, 2)
        last_character = file.read(1)

    if last_character not in (b"\n", b"\r"):
        with HISTORY_PATH.open("ab") as file:
            file.write(b"\n")


def load_prediction_history():
    initialize_history_file()

    try:
        history_df = pd.read_csv(
            HISTORY_PATH,
            encoding="utf-8-sig",
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    for column in HISTORY_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = pd.NA

    history_df = history_df[
        HISTORY_COLUMNS
    ].copy()

    history_df["PredictionTime"] = pd.to_datetime(
        history_df["PredictionTime"],
        errors="coerce",
    )

    history_df["LastPurchaseDate"] = pd.to_datetime(
        history_df["LastPurchaseDate"],
        errors="coerce",
    )

    for column in [
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
    ]:
        history_df[column] = pd.to_numeric(
            history_df[column],
            errors="coerce",
        )

    history_df = history_df.dropna(
        subset=[
            "CustomerID",
            "PredictionTime",
        ]
    )

    history_df["CustomerID"] = (
        history_df["CustomerID"]
        .astype(str)
        .str.strip()
    )

    return history_df


def append_prediction_history(records_df):
    """
    Thêm lịch sử Single hoặc Batch vào CSV.
    """
    initialize_history_file()
    ensure_file_ends_with_newline()

    records_df = records_df.copy()

    for column in HISTORY_COLUMNS:
        if column not in records_df.columns:
            records_df[column] = pd.NA

    records_df = records_df[
        HISTORY_COLUMNS
    ]

    records_df.to_csv(
        HISTORY_PATH,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def clear_prediction_history():
    pd.DataFrame(
        columns=HISTORY_COLUMNS
    ).to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )