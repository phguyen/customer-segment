import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS dùng chung cho toàn bộ ứng dụng
st.markdown(
    """
    <style>
        /* Khoảng cách nội dung chính */
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Tiêu đề các trang */
        h1 {
            color: #172033;
            font-weight: 750;
            letter-spacing: -0.5px;
        }

        h2, h3 {
            color: #253047;
        }

        /* Nút bấm */
        div.stButton > button,
        div.stFormSubmitButton > button {
            min-height: 46px;
            border-radius: 10px;
            font-weight: 650;
            border: none;
        }

        /* Các ô nhập liệu */
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {
            border-radius: 10px;
        }

        /* File uploader */
        section[data-testid="stFileUploaderDropzone"] {
            border-radius: 14px;
            border: 1.5px dashed #8BA9E8;
            background-color: #F8FAFF;
        }

        /* Metric */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5EAF3;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(31, 42, 68, 0.05);
        }

        /* Bỏ bớt khoảng trống trên sidebar */
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            padding-top: 1rem;
        }

        /* Khối giới thiệu ở sidebar */
        .sidebar-brand {
            padding: 10px 4px 18px 4px;
        }

        .sidebar-title {
            font-size: 1.12rem;
            font-weight: 750;
            color: #172033;
            margin-bottom: 4px;
        }

        .sidebar-subtitle {
            font-size: 0.82rem;
            line-height: 1.45;
            color: #64748B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

dashboard_page = st.Page(
    "pages/dashboard.py",
    title="Tổng quan",
    icon=":material/dashboard:",
    default=True,
)

single_page = st.Page(
    "pages/single_prediction.py",
    title="Phân nhóm đơn lẻ",
    icon=":material/person_search:",
)

batch_page = st.Page(
    "pages/batch_prediction.py",
    title="Phân nhóm hàng loạt",
    icon=":material/upload_file:",
)

model_page = st.Page(
    "pages/model_information.py",
    title="Thông tin mô hình",
    icon=":material/model_training:",
)

history_page = st.Page(
    "pages/prediction_history.py",
    title="Lịch sử phân tích",
    icon=":material/history:",
)

navigation = st.navigation([
    dashboard_page,
    single_page,
    batch_page,
    model_page,
    history_page,
])

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">👥 Customer Segmentation</div>
            <div class="sidebar-subtitle">
                Hệ thống phân tích và phân khúc khách hàng dựa trên mô hình RFM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("TRẠNG THÁI HỆ THỐNG")
    st.success("Đang hoạt động")

    

navigation.run()