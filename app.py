
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Hệ thống Đánh giá Rủi ro Tín dụng",
    page_icon="🏦",
    layout="centered",
)

BUNDLE_PATH = Path("credit_risk_pipeline.pkl")

FEATURE_LABELS = {
    "CREDIT_CARD_NUMBER_OF_LATE_PAYMENT": "Số lần thanh toán thẻ tín dụng trễ hạn",
    "ENQUIRIES_3M": "Số lần tra cứu tín dụng trong 3 tháng",
    "ENQUIRIES_12M": "Số lần tra cứu tín dụng trong 12 tháng",
    "SHORT_TERM_COUNT": "Số khoản vay ngắn hạn",
    "OUTSTANDING_BAL_ALL_CURRENT": "Tổng dư nợ hiện tại (VND)",
    "NUM_NEW_LOAN_TAKEN_3M": "Số khoản vay mới trong 3 tháng",
    "NUMBER_OF_LOANS": "Tổng số khoản vay",
    "CREDIT_CARD_MONTH_SINCE_30DPD_FLAG": (
        "Có lịch sử quá hạn thẻ tín dụng 30 ngày (0: Không, 1: Có)"
    ),
    "CREDIT_CARD_MONTH_SINCE_90DPD_FLAG": (
        "Có lịch sử quá hạn thẻ tín dụng 90 ngày (0: Không, 1: Có)"
    ),
    "CC_TO_ALL_LOAN_RATIO": "Tỷ lệ thẻ tín dụng trên tổng khoản vay",
    "SHORT_TERM_RATIO": "Tỷ lệ khoản vay ngắn hạn",
}

DPD_RAW_COLS = [
    "CREDIT_CARD_MONTH_SINCE_10DPD",
    "CREDIT_CARD_MONTH_SINCE_30DPD",
    "CREDIT_CARD_MONTH_SINCE_60DPD",
    "CREDIT_CARD_MONTH_SINCE_90DPD",
]


@st.cache_resource
def load_bundle(bundle_path: str, modified_time: float):
    """modified_time giúp Streamlit tự nạp lại khi thay file model."""
    return joblib.load(bundle_path)


if not BUNDLE_PATH.exists():
    st.error(
        "Không tìm thấy `credit_risk_pipeline.pkl`. "
        "Hãy chạy Cell 11 và đặt file này cùng thư mục với `app.py`."
    )
    st.stop()

bundle = load_bundle(
    str(BUNDLE_PATH),
    BUNDLE_PATH.stat().st_mtime,
)

model = bundle["model"]
features = bundle["selected_features"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
winsor_bounds = bundle["winsor_bounds"]
balance_offset = bundle["balance_offset"]
dpd_sentinel = bundle["dpd_sentinel"]
bad_label = bundle["bad_label"]
decision_threshold = bundle["decision_threshold"]
input_defaults = bundle["input_defaults"]


def friendly_feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def clean_base_input(data: pd.DataFrame) -> pd.DataFrame:
    """Tái tạo các bước làm sạch cố định đã dùng khi train."""
    cleaned = data.copy()

    # Trừ offset cho các biến dư nợ.
    for col in cleaned.columns:
        if "OUTSTANDING_BAL" in col:
            cleaned[col] = pd.to_numeric(
                cleaned[col],
                errors="coerce",
            ) - balance_offset

    # Hỗ trợ nếu model có dùng các cột DPD thô.
    for col in DPD_RAW_COLS:
        if col not in cleaned.columns:
            continue

        original_values = cleaned[col]
        flag_col = f"{col}_FLAG"

        if flag_col in features:
            cleaned[flag_col] = (
                original_values.notna()
                & original_values.ne(dpd_sentinel)
            ).astype(float)

        cleaned[col] = original_values.mask(
            original_values.eq(dpd_sentinel),
            pd.NA,
        )

    return cleaned


def apply_winsorization(data: pd.DataFrame) -> pd.DataFrame:
    """Dùng đúng ngưỡng outlier đã học từ tập train."""
    transformed = data.copy()

    for col, (lower, upper) in winsor_bounds.items():
        if col in transformed.columns:
            transformed[col] = transformed[col].clip(lower, upper)

    return transformed


def preprocess_for_model(user_input: dict) -> pd.DataFrame:
    """Đầu vào người dùng → dữ liệu đúng định dạng model đã train."""
    input_df = pd.DataFrame([user_input]).reindex(columns=features)

    cleaned_df = clean_base_input(input_df)
    winsorized_df = apply_winsorization(cleaned_df)

    # Thứ tự cột phải khớp chính xác thứ tự lúc train.
    model_input = winsorized_df.reindex(columns=features)

    imputed_values = imputer.transform(model_input)
    scaled_values = scaler.transform(imputed_values)

    return pd.DataFrame(
        scaled_values,
        columns=features,
    )


def set_profile(overrides: dict):
    """Cập nhật hồ sơ mẫu trước khi các input widget được tạo."""
    for feature in features:
        key = f"input_{feature}"
        default_value = float(input_defaults.get(feature, 0.0))
        st.session_state[key] = float(
            overrides.get(feature, default_value)
        )


# Giá trị này chỉ là minh họa giao diện.
# Quyết định cuối cùng luôn do model dự đoán.
safe_profile = {
    "CREDIT_CARD_NUMBER_OF_LATE_PAYMENT": 0,
    "ENQUIRIES_3M": 1,
    "ENQUIRIES_12M": 2,
    "SHORT_TERM_COUNT": 1,
    "OUTSTANDING_BAL_ALL_CURRENT": 10_000_000,
    "NUM_NEW_LOAN_TAKEN_3M": 0,
    "NUMBER_OF_LOANS": 2,
    "CREDIT_CARD_MONTH_SINCE_30DPD_FLAG": 0,
    "CREDIT_CARD_MONTH_SINCE_90DPD_FLAG": 0,
    "CC_TO_ALL_LOAN_RATIO": 0.20,
    "SHORT_TERM_RATIO": 0.30,
}

risk_profile = {
    "CREDIT_CARD_NUMBER_OF_LATE_PAYMENT": 10,
    "ENQUIRIES_3M": 15,
    "ENQUIRIES_12M": 35,
    "SHORT_TERM_COUNT": 10,
    "OUTSTANDING_BAL_ALL_CURRENT": 500_000_000,
    "NUM_NEW_LOAN_TAKEN_3M": 8,
    "NUMBER_OF_LOANS": 15,
    "CREDIT_CARD_MONTH_SINCE_30DPD_FLAG": 1,
    "CREDIT_CARD_MONTH_SINCE_90DPD_FLAG": 1,
    "CC_TO_ALL_LOAN_RATIO": 0.90,
    "SHORT_TERM_RATIO": 0.95,
}


# Khởi tạo session state
for feature in features:
    key = f"input_{feature}"

    if key not in st.session_state:
        st.session_state[key] = float(
            input_defaults.get(feature, 0.0)
        )


st.title("🏦 HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG")

st.write(
    f"""
    Hệ thống sử dụng mô hình **{bundle["model_name"]}** để ước tính
    xác suất khách hàng thuộc nhóm **nợ xấu**.

    Ngưỡng từ chối hiện tại: **{decision_threshold:.0%}**.
    """
)

st.divider()

st.subheader("⚡ Hồ sơ mẫu")

col_safe, col_risk = st.columns(2)

with col_safe:
    if st.button("🟢 Hồ sơ rủi ro thấp", use_container_width=True):
        set_profile(safe_profile)

with col_risk:
    if st.button("🔴 Hồ sơ rủi ro cao", use_container_width=True):
        set_profile(risk_profile)

st.divider()
st.subheader("📄 Thông tin khách hàng")

user_input = {}

for feature in features:
    widget_key = f"input_{feature}"

    is_count_like = any(
        text in feature
        for text in [
            "COUNT",
            "NUMBER_OF",
            "ENQUIRIES",
            "NUM_NEW_LOAN",
            "_FLAG",
        ]
    )

    user_input[feature] = st.number_input(
        label=friendly_feature_name(feature),
        min_value=0.0,
        step=1.0 if is_count_like else 0.01,
        format="%.0f" if is_count_like else "%.4f",
        key=widget_key,
    )

st.divider()

if st.button("🚀 PHÂN TÍCH", use_container_width=True):
    try:
        X_input = preprocess_for_model(user_input)

        probabilities = model.predict_proba(X_input)[0]
        bad_index = list(model.classes_).index(bad_label)
        bad_probability = float(probabilities[bad_index])

        st.subheader("📊 KẾT QUẢ")
        st.metric("Xác suất nợ xấu", f"{bad_probability:.2%}")
        st.progress(int(round(bad_probability * 100)))

        if bad_probability >= decision_threshold:
            st.error("### 🔴 TỪ CHỐI CẤP TÍN DỤNG")
            st.write(
                "Khách hàng thuộc nhóm **rủi ro cao** "
                "theo ngưỡng quyết định hiện tại."
            )
        else:
            st.success("### 🟢 PHÊ DUYỆT HỒ SƠ")
            st.write(
                "Khách hàng thuộc nhóm **rủi ro thấp** "
                "theo ngưỡng quyết định hiện tại."
            )

    except Exception as error:
        st.error("Không thể phân tích hồ sơ.")
        st.exception(error)

