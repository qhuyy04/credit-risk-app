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
LOOKUP_PATH = Path("customer_lookup.csv")


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error(f"Không tìm thấy file model: {BUNDLE_PATH}")
        st.stop()
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_lookup():
    if not LOOKUP_PATH.exists():
        st.error(f"Không tìm thấy file dữ liệu khách hàng: {LOOKUP_PATH}")
        st.stop()
    df = pd.read_csv(LOOKUP_PATH)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df.set_index("customer_id")


bundle = load_bundle()
lookup_df = load_lookup()

model = bundle["model"]
selected_features = bundle["selected_features"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
winsor_bounds = bundle["winsor_bounds"]
bad_label = bundle["bad_label"]
decision_threshold = bundle.get("decision_threshold", 0.5)


def predict_risk(row: pd.Series) -> float:
    """Trả về xác suất khách hàng rủi ro (nợ xấu), dùng đúng pipeline lúc train."""
    input_df = pd.DataFrame([row[selected_features]])

    for col, (lower, upper) in winsor_bounds.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].clip(lower, upper)

    input_df = input_df.reindex(columns=selected_features)
    X = scaler.transform(imputer.transform(input_df))
    proba = model.predict_proba(X)[0]
    bad_index = list(model.classes_).index(bad_label)
    return float(proba[bad_index])


def fmt_money(value) -> str:
    if pd.isna(value):
        return "Không có dữ liệu"
    return f"{value:,.0f} VNĐ".replace(",", ".")


def fmt_count(value) -> str:
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}".replace(",", ".")


st.title("🏦 Hệ thống Đánh giá Rủi ro Tín dụng")
st.caption("Tra cứu hồ sơ khách hàng theo mã khách hàng và đánh giá khả năng rủi ro.")

st.divider()
st.subheader("🔍 Tra cứu khách hàng")

with st.form("lookup_form"):
    customer_id_input = st.text_input("Mã khách hàng (ID)", placeholder="VD: 2911")
    submitted = st.form_submit_button("🔎 Tra cứu", use_container_width=True)

if submitted:
    if not customer_id_input.strip().isdigit():
        st.error("Mã khách hàng phải là số. Vui lòng nhập lại.")
        st.stop()

    customer_id = int(customer_id_input.strip())

    if customer_id not in lookup_df.index:
        st.error(f"Không tìm thấy khách hàng với mã **{customer_id}** trong hệ thống.")
        st.stop()

    row = lookup_df.loc[customer_id]

    st.divider()
    st.subheader("📋 Thông tin khách hàng")
    st.caption("Dữ liệu do hệ thống tự động tra cứu, không thể chỉnh sửa.")

    st.markdown(f"**Mã khách hàng:** {customer_id}")

    st.markdown("##### Tổng quan khoản vay")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng số khoản vay", fmt_count(row["NUMBER_OF_LOANS"]))
    m2.metric("Vay ngắn hạn", fmt_count(row["SHORT_TERM_COUNT"]))
    m3.metric("Vay trung hạn", fmt_count(row["MID_TERM_COUNT"]))
    m4.metric("Vay dài hạn", fmt_count(row["LONG_TERM_COUNT"]))

    bal1, bal2 = st.columns(2)
    bal1.metric("Tổng dư nợ hiện tại (vay + thẻ)", fmt_money(row["OUTSTANDING_BAL_ALL_CURRENT"]))
    bal2.metric("Dư nợ vay hiện tại", fmt_money(row["OUTSTANDING_BAL_LOAN_CURRENT"]))

    st.markdown("##### Khoản vay mới phát sinh")
    new_loan_table = pd.DataFrame(
        {
            "Kỳ hạn": ["3 tháng", "6 tháng", "9 tháng", "12 tháng"],
            "Tổng": [
                fmt_count(row["NUM_NEW_LOAN_TAKEN_3M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_6M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_9M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_12M"]),
            ],
            "Từ ngân hàng": [
                fmt_count(row["NUM_NEW_LOAN_TAKEN_BANK_3M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_BANK_6M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_BANK_9M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_BANK_12M"]),
            ],
            "Từ phi ngân hàng": [
                fmt_count(row["NUM_NEW_LOAN_TAKEN_NON_BANK_3M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_NON_BANK_6M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_NON_BANK_9M"]),
                fmt_count(row["NUM_NEW_LOAN_TAKEN_NON_BANK_12M"]),
            ],
        }
    )
    st.dataframe(new_loan_table, hide_index=True, use_container_width=True)

    st.markdown("##### Thẻ tín dụng")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Tổng số thẻ tín dụng", fmt_count(row["NUMBER_OF_CREDIT_CARDS"]))
    cc2.metric("Từ ngân hàng", fmt_count(row["NUMBER_OF_CREDIT_CARDS_BANK"]))
    cc3.metric("Từ phi ngân hàng", fmt_count(row["NUMBER_OF_CREDIT_CARDS_NON_BANK"]))

    cc_bal_table = pd.DataFrame(
        {
            "Thời điểm": ["Hiện tại", "3 tháng trước", "6 tháng trước", "9 tháng trước", "12 tháng trước"],
            "Dư nợ thẻ tín dụng": [
                fmt_money(row["OUTSTANDING_BAL_CC_CURRENT"]),
                fmt_money(row["OUTSTANDING_BAL_CC_3M"]),
                fmt_money(row["OUTSTANDING_BAL_CC_6M"]),
                fmt_money(row["OUTSTANDING_BAL_CC_9M"]),
                fmt_money(row["OUTSTANDING_BAL_CC_12M"]),
            ],
        }
    )
    st.dataframe(cc_bal_table, hide_index=True, use_container_width=True)

    st.markdown("##### Quan hệ tín dụng")
    r1, r2, r3 = st.columns(3)
    r1.metric("Tổng số mối quan hệ", fmt_count(row["NUMBER_OF_RELATIONSHIP"]))
    r2.metric("Với ngân hàng", fmt_count(row["NUMBER_OF_RELATIONSHIP_BANK"]))
    r3.metric("Với phi ngân hàng", fmt_count(row["NUMBER_OF_RELATIONSHIP_NON_BANK"]))

    st.markdown("##### Tra cứu tín dụng chi tiết")
    enq_table = pd.DataFrame(
        {
            "Kỳ hạn": ["3 tháng", "6 tháng", "9 tháng", "12 tháng"],
            "Tổng lượt tra cứu": [
                fmt_count(row["ENQUIRIES_3M"]),
                fmt_count(row["ENQUIRIES_6M"]),
                fmt_count(row["ENQUIRIES_9M"]),
                fmt_count(row["ENQUIRIES_12M"]),
            ],
            "Tra cứu vay": [
                fmt_count(row["ENQUIRIES_FOR_LOAN_3M"]),
                fmt_count(row["ENQUIRIES_FOR_LOAN_6M"]),
                fmt_count(row["ENQUIRIES_FOR_LOAN_9M"]),
                fmt_count(row["ENQUIRIES_FOR_LOAN_12M"]),
            ],
            "Tra cứu thẻ": [
                fmt_count(row["ENQUIRIES_FOR_CC_3M"]),
                fmt_count(row["ENQUIRIES_FOR_CC_6M"]),
                fmt_count(row["ENQUIRIES_FOR_CC_9M"]),
                fmt_count(row["ENQUIRIES_FOR_CC_12M"]),
            ],
            "Từ ngân hàng": [
                fmt_count(row["ENQUIRIES_FROM_BANK_3M"]),
                fmt_count(row["ENQUIRIES_FROM_BANK_6M"]),
                fmt_count(row["ENQUIRIES_FROM_BANK_9M"]),
                fmt_count(row["ENQUIRIES_FROM_BANK_12M"]),
            ],
            "Từ phi ngân hàng": [
                fmt_count(row["ENQUIRIES_FROM_NON_BANK_3M"]),
                fmt_count(row["ENQUIRIES_FROM_NON_BANK_6M"]),
                fmt_count(row["ENQUIRIES_FROM_NON_BANK_9M"]),
                fmt_count(row["ENQUIRIES_FROM_NON_BANK_12M"]),
            ],
        }
    )
    st.dataframe(enq_table, hide_index=True, use_container_width=True)

    st.markdown("##### Lịch sử trễ hạn thanh toán (thẻ tín dụng)")

    def fmt_dpd(months, flag):
        if pd.isna(flag):
            return "Không có dữ liệu"
        if flag == 0:
            return "Chưa từng trễ hạn"
        if pd.isna(months):
            return "Có trễ hạn (không rõ số tháng)"
        return f"Trễ hạn — cách đây {months:.0f} tháng"

    dpd_table = pd.DataFrame(
        {
            "Mức độ trễ hạn": ["10 ngày", "30 ngày", "60 ngày", "90 ngày"],
            "Tình trạng": [
                fmt_dpd(row["CREDIT_CARD_MONTH_SINCE_10DPD"], row["CREDIT_CARD_MONTH_SINCE_10DPD_FLAG"]),
                fmt_dpd(row["CREDIT_CARD_MONTH_SINCE_30DPD"], row["CREDIT_CARD_MONTH_SINCE_30DPD_FLAG"]),
                fmt_dpd(row["CREDIT_CARD_MONTH_SINCE_60DPD"], row["CREDIT_CARD_MONTH_SINCE_60DPD_FLAG"]),
                fmt_dpd(row["CREDIT_CARD_MONTH_SINCE_90DPD"], row["CREDIT_CARD_MONTH_SINCE_90DPD_FLAG"]),
            ],
        }
    )
    st.dataframe(dpd_table, hide_index=True, use_container_width=True)
    st.metric("Tổng số lần trễ hạn thanh toán", fmt_count(row["CREDIT_CARD_NUMBER_OF_LATE_PAYMENT"]))

    st.divider()
    st.subheader("⚖️ Đánh giá rủi ro")

    probability = predict_risk(row)
    is_high_risk = probability >= decision_threshold

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.metric("Xác suất rủi ro (nợ xấu)", f"{probability * 100:.1f}%")
    with risk_col2:
        st.metric("Ngưỡng quyết định", f"{decision_threshold * 100:.0f}%")

    if is_high_risk:
        st.error(
            "🔴 **KHÁCH HÀNG RỦI RO CAO** — Không nên cấp thêm khoản vay mới. "
            "Cần xem xét kỹ hồ sơ và lịch sử tín dụng trước khi ra quyết định."
        )
    else:
        st.success(
            "🟢 **KHÁCH HÀNG RỦI RO THẤP** — Có thể xem xét cấp khoản vay theo "
            "quy trình thẩm định thông thường."
        )

    st.caption(
        "Đánh giá dựa trên mô hình học máy huấn luyện từ dữ liệu lịch sử tín dụng. "
        "Kết quả chỉ mang tính tham khảo, không thay thế cho quy trình thẩm định chính thức."
    )

st.divider()
st.caption("Hệ thống nội bộ — chỉ dùng cho mục đích tra cứu và hỗ trợ ra quyết định.")
