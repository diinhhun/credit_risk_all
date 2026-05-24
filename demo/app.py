"""
BƯỚC 7 – STREAMLIT WEB APP
=============================================================
File này là giao diện web cuối cùng của hệ thống phê duyệt tín dụng.
Nó tích hợp toàn bộ pipeline đã xây dựng ở các bước trước:
  - Bước 2: preprocessor.pkl  (Pipeline tiền xử lý dữ liệu)
  - Bước 4: xgb_model.pkl     (Mô hình XGBoost dự đoán)
  - Bước 5: kmeans_model.pkl  (Mô hình phân cụm K-Means)
  - Bước 6: pca_model.pkl     (Giảm chiều PCA – dùng cho K-Means)

Cách chạy: streamlit run app.py
=============================================================
"""
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# ĐỊNH NGHĨA LẠI CLASS KMEANS TỰ CÀI ĐẶT
# =============================================================
class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k; self.max_iters = max_iters; self.tol = tol
        self.random_state = random_state; self.centroids = None

    def fit(self, X):
        np.random.seed(self.random_state)
        idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[idx]
        for i in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else self.centroids[j] for j in range(self.k)])
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol): break
            self.centroids = new_centroids
        self.labels_ = labels; return self

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            distances[:, j] = np.linalg.norm(X - self.centroids[j], axis=1)
        return distances

# =============================================================
# CẤU HÌNH GIAO DIỆN
# =============================================================
st.set_page_config(page_title="Credit Risk App", layout="wide")
st.title("🏦 Hệ Thống Phê Duyệt Tín Dụng (Credit Risk Approval System)")

# =============================================================
# CÁC HẰNG SỐ (CONSTANTS)
# =============================================================
MAX_EMP_TRAIN     = 60.0        # person_emp_length max hợp lệ
MAX_RATE_TRAIN    = 24.0        # loan_int_rate max trong tập train
MIN_RATE_TRAIN    = 5.0         # loan_int_rate min
DTI_HARD_LIMIT    = 0.50        # Tỷ lệ vay/thu nhập > 50% → auto từ chối
RISK_THRESHOLD    = 0.35        # Ngưỡng xác suất phê duyệt

# =============================================================
# TẢI CÁC MÔ HÌNH
# =============================================================
@st.cache_resource
def load_models():
    try:
        pp = joblib.load("preprocessor.pkl")
    except FileNotFoundError:
        st.error("Không tìm thấy file preprocessor.pkl.")
        st.stop()
        
    try:
        xgb = joblib.load("xgb_model.pkl")
    except FileNotFoundError:
        st.error("Không tìm thấy file xgb_model.pkl.")
        st.stop()
        
    try:
        km = joblib.load("kmeans_model.pkl")
    except FileNotFoundError:
        km = None
        
    try:
        pca = joblib.load("pca_model.pkl") 
    except FileNotFoundError:
        pca = None
        
    try:
        explainer = shap.TreeExplainer(xgb)
    except Exception:
        explainer = None
        
    return pp, xgb, km, pca, explainer

preprocessor, xgb_model, kmeans_model, pca_model, shap_explainer = load_models()

# =============================================================
# KHAI BÁO FEATURES
# =============================================================
num_feats = ['person_age','person_income','person_emp_length',
             'loan_amnt','loan_int_rate','loan_percent_income','cb_person_default_on_file']

ohe_feats = list(preprocessor.named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(['person_home_ownership','loan_intent']))

all_feats = num_feats + ohe_feats

feat_vi = {
    'person_age':'Tuổi', 'person_income':'Thu nhập', 'person_emp_length':'Thâm niên (năm)',
    'loan_amnt':'Số tiền vay', 'loan_int_rate':'Lãi suất (%)', 'loan_percent_income':'Tỷ lệ vay/TN',
    'cb_person_default_on_file':'Lịch sử vỡ nợ',
    'person_home_ownership_MORTGAGE':'Nhà: Thế chấp','person_home_ownership_OTHER':'Nhà: Khác',
    'person_home_ownership_OWN':'Nhà: Sở hữu','person_home_ownership_RENT':'Nhà: Thuê',
    'loan_intent_DEBTCONSOLIDATION':'Vay: Gộp nợ','loan_intent_EDUCATION':'Vay: Học tập',
    'loan_intent_HOMEIMPROVEMENT':'Vay: Sửa nhà','loan_intent_MEDICAL':'Vay: Y tế',
    'loan_intent_PERSONAL':'Vay: Cá nhân','loan_intent_VENTURE':'Vay: Kinh doanh',
}

# =============================================================
# SIDEBAR – GIAO DIỆN NHẬP LIỆU
# =============================================================
st.sidebar.header("📋 Thông tin khách hàng")

age    = st.sidebar.number_input("Age (Tuổi)", 18, 100, 30)
income = st.sidebar.number_input("Annual Income – USD", 0, 10_000_000, 50000, 1000)
if income <= 0: income = 1

_HOME_OPTS  = {"RENT": "RENT (Thuê)", "MORTGAGE": "MORTGAGE (Thế chấp)", "OWN": "OWN (Sở hữu)", "OTHER": "OTHER (Khác)"}
home_val   = st.sidebar.selectbox("Home Ownership", list(_HOME_OPTS.keys()), format_func=lambda x: _HOME_OPTS[x])

emp    = st.sidebar.number_input("Employment Length – years", 0.0, None, 5.0, 0.5)

_DEF_OPTS = {0: "Không (No)", 1: "Có (Yes)"}
default_val = st.sidebar.selectbox("Lịch sử vỡ nợ trước đây", list(_DEF_OPTS.keys()), format_func=lambda x: _DEF_OPTS[x])

st.sidebar.header("💰 Thông tin khoản vay")

_INTENT_OPTS = {
    "PERSONAL": "PERSONAL (Cá nhân)", "EDUCATION": "EDUCATION (Học tập)",
    "MEDICAL": "MEDICAL (Y tế)", "VENTURE": "VENTURE (Kinh doanh)",
    "HOMEIMPROVEMENT": "HOMEIMPROVEMENT (Sửa nhà)", "DEBTCONSOLIDATION": "DEBTCONSOLIDATION (Gộp nợ)",
}
intent_val  = st.sidebar.selectbox("Loan Intent", list(_INTENT_OPTS.keys()), format_func=lambda x: _INTENT_OPTS[x])

amount  = st.sidebar.number_input("Loan Amount – USD", 500, 10_000_000, 10000, 500)
rate    = st.sidebar.slider("Interest Rate %", 5.0, 25.0, 11.0, 0.1)

# Tự động tính Tỷ lệ vay/thu nhập
pct = amount / income
dti_color = "🔴" if pct > DTI_HARD_LIMIT else "🟢"
st.sidebar.metric("Tỷ lệ Vay / Thu nhập", f"{pct:.2%}")
if pct > DTI_HARD_LIMIT:
    st.sidebar.error(f"⚠️ DTI {pct:.1%} > {DTI_HARD_LIMIT:.0%} – Sẽ tự động từ chối")

run = st.sidebar.button("Thẩm định hồ sơ (Run)")

# =============================================================
# LOGIC NGHIỆP VỤ
# =============================================================
def apply_business_rules(income, loan_amnt, loan_pct):
    if loan_pct > DTI_HARD_LIMIT:
        return True, (
            f"Tỷ lệ vay/thu nhập {loan_pct:.1%} vượt ngưỡng tối đa {DTI_HARD_LIMIT:.0%}. "
            "Ngân hàng không cho vay khi tỷ lệ nợ/thu nhập vượt 50%."
        ), 'DTI_EXCEEDED'
    return False, "", ""

def clip_to_training_bounds(df):
    df = df.copy()
    clip_map = {
        'loan_percent_income': (0.0,         DTI_HARD_LIMIT),
        'person_emp_length':   (0.0,         MAX_EMP_TRAIN),
        'loan_int_rate':       (MIN_RATE_TRAIN, MAX_RATE_TRAIN),
        'person_age':          (18,          100),
    }
    for col, (lo, hi) in clip_map.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df

# =============================================================
# KHỐI DỰ ĐOÁN CHÍNH
# =============================================================
if run:
    if emp > (age - 18):
        st.warning(f"⚠️ Khách hàng {age} tuổi không thể có {emp} năm thâm niên làm việc. (Giả định tuổi bắt đầu lao động tối thiểu là 18, nên thâm niên tối đa chỉ có thể là {age - 18} năm). Vui lòng kiểm tra lại!")
        st.stop()

    input_df = pd.DataFrame([{
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_val,
        'person_emp_length': emp,
        'loan_intent': intent_val,
        'loan_amnt': amount,
        'loan_int_rate': rate,
        'loan_percent_income': pct,
        'cb_person_default_on_file': default_val
    }])

    rejected, msg, rule_code = apply_business_rules(income, amount, pct)
    
    if rejected:
        st.error("🚫 **Kết quả thẩm định: TỪ CHỐI HỒ SƠ VAY (BUSINESS RULES)**")
        st.divider()

        monthly_income = income / 12
        monthly_payment = (amount * rate / 100) / 12
        max_eligible = round(income * DTI_HARD_LIMIT)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            label="📊 Tỷ lệ Nợ/Thu nhập (DTI)",
            value=f"{pct:.1%}",
            delta=f"Vượt ngưỡng {DTI_HARD_LIMIT:.0%}",
            delta_color="inverse"
        )
        col_b.metric(
            label="🏦 Khoản vay tối đa đủ điều kiện",
            value=f"${max_eligible:,.0f}",
            delta=f"Khách đề nghị ${amount:,.0f}"
        )
        col_c.metric(
            label="💳 Lãi ước tính/tháng",
            value=f"${monthly_payment:,.0f}",
            delta=f"Thu nhập/tháng ${monthly_income:,.0f}"
        )

        st.divider()
        st.subheader("📋 Lý do từ chối")
        st.markdown(f"""
| Tiêu chí | Giá trị khách hàng | Tiêu chuẩn ngân hàng | Đánh giá |
|---|---|---|---|
| Số tiền vay | ${amount:,.0f} | Tối đa ${max_eligible:,.0f} | ❌ Vượt ${amount-max_eligible:,.0f} |
| Thu nhập năm | ${income:,.0f} | — | ✅ |
| **Tỷ lệ Nợ/Thu nhập (DTI)** | **{pct:.1%}** | **Tối đa {DTI_HARD_LIMIT:.0%}** | **❌ Vượt {(pct - DTI_HARD_LIMIT)*100:.1f}%** |
| Lãi ước tính/tháng | ${monthly_payment:,.0f} | ≤ ${monthly_income*DTI_HARD_LIMIT:,.0f} | ❌ Quá cao |

> ⚠️ **DTI {pct:.1%} vượt ngưỡng an toàn {DTI_HARD_LIMIT:.0%}.** Nếu vay ${amount:,.0f} với lãi suất {rate:.1f}%/năm, khách hàng phải trả **${monthly_payment:,.0f}/tháng** trong khi thu nhập tháng chỉ đạt **${monthly_income:,.0f}** – không đảm bảo khả năng trả nợ.
""")
        st.stop()

    input_df_clipped = clip_to_training_bounds(input_df)
    X_proc = preprocessor.transform(input_df_clipped)

    prob = xgb_model.predict_proba(X_proc)[0]
    pred = int(prob[1] >= RISK_THRESHOLD)

    col1, col2 = st.columns(2)

    with col1:
        if pred == 0:
            st.success(f"✅ PHÊ DUYỆT (APPROVED)\n\nXác suất trả nợ đúng hạn: **{prob[0]:.1%}**")
        else:
            st.error(f"❌ TỪ CHỐI (REJECTED)\n\nXác suất vỡ nợ: **{prob[1]:.1%}**")

        st.progress(float(prob[1]), text=f"Default risk (Rủi ro vỡ nợ): {prob[1]:.1%}")
        st.caption(f"Ngưỡng phán quyết: {RISK_THRESHOLD:.0%} | Xác suất vỡ nợ hiện tại: {prob[1]:.1%}")

    with col2:
        st.subheader("Hồ sơ (Profile)")
        st.table(pd.DataFrame({
            "Chỉ tiêu (Field)": ["Age (Tuổi)","Income (Thu nhập)","Loan (Vay)","Rate (Lãi suất)","Ratio (Tỷ lệ vay/TN)"],
            "Giá trị (Value)":  [f"{age}", f"${income:,}", f"${amount:,}", f"{rate}%", f"{pct:.0%}"]
        }))

    st.divider()

    st.subheader("🧠 Giải thích AI – SHAP (AI Explanation)")
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(X_proc)[0]
            order = np.argsort(np.abs(sv))[::-1]

            # ── Top 3 lý do (style bước 6) ──────────────────────────────
            if pred == 1:
                st.markdown("**Top 3 lý do từ chối (Top 3 Rejection Reasons):**")
                count = 0
                for i in order:
                    if sv[i] > 0 and count < 3:
                        count += 1
                        ten = feat_vi.get(all_feats[i], all_feats[i])
                        raw_val = input_df_clipped.iloc[0].get(
                            all_feats[i] if all_feats[i] in input_df_clipped.columns else None, X_proc[0][i]
                        )
                        st.markdown(
                            f"**Lý do {count}: {ten}**  \n"
                            f"&emsp;Giá trị: `{X_proc[0][i]:.2f}` &nbsp;|&nbsp; SHAP: `+{sv[i]:.4f}` → Tăng rủi ro vỡ nợ 🔴"
                        )
            else:
                st.markdown("**Top 3 yếu tố hỗ trợ phê duyệt (Top 3 Approval Factors):**")
                count = 0
                for i in order:
                    if sv[i] < 0 and count < 3:
                        count += 1
                        ten = feat_vi.get(all_feats[i], all_feats[i])
                        st.markdown(
                            f"**Yếu tố {count}: {ten}**  \n"
                            f"&emsp;Giá trị: `{X_proc[0][i]:.2f}` &nbsp;|&nbsp; SHAP: `{sv[i]:.4f}` → Giảm rủi ro vỡ nợ 🟢"
                        )

            st.divider()

            # ── Waterfall plot (style bước 6) ────────────────────────────
            st.markdown("**📊 Biểu đồ Waterfall – Đóng góp từng yếu tố (SHAP Waterfall Plot):**")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.sca(ax)

            explanation = shap.Explanation(
                values      = sv,
                base_values = shap_explainer.expected_value,
                data        = X_proc[0],
                feature_names=[feat_vi.get(f, f) for f in all_feats]
            )
            shap.waterfall_plot(explanation, max_display=10, show=False)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP error: {e}")
    else:
        st.warning("Mô hình SHAP không khởi tạo được. Vui lòng kiểm tra lại môi trường thư viện.")

    st.divider()
    st.subheader("🗂️ Phân cụm rủi ro (Risk Cluster – K-Means)")

    cluster_label = None
    if kmeans_model is not None and pca_model is not None:
        try:
            X_pca_input = pca_model.transform(X_proc)
            cluster_id = int(kmeans_model.predict(X_pca_input)[0])
            cluster_label = cluster_id
        except Exception:
            cluster_label = None

    p = prob[1]

    if cluster_label is not None:
        cluster_desc = {
            0: ("success", "**Cluster 0 – Rủi ro Thấp (Low Risk)** – Hồ sơ tốt, thu nhập ổn định, lịch sử tín dụng tốt"),
            1: ("warning", "**Cluster 1 – Rủi ro Trung bình (Medium Risk)** – Cần xem xét thêm thông tin"),
            2: ("error",   "**Cluster 2 – Rủi ro Cao (High Risk)** – Nguy cơ vỡ nợ cao, thâm niên thấp hoặc thu nhập thấp"),
        }
        kind, msg = cluster_desc.get(cluster_label, ("warning", f"**Cluster {cluster_label}** – Không xác định"))
        
        if kind == "success":
            st.success(f"Cluster thực tế (KMeans): {msg}")
        elif kind == "warning":
            st.warning(f"Cluster thực tế (KMeans): {msg}")
        else:
            st.error(f"Cluster thực tế (KMeans): {msg}")
    else:
        if p < 0.25:
            st.success("Cluster (ước tính): **Rủi ro Thấp (Low Risk)** – Hồ sơ tốt")
        elif p < 0.55:
            st.warning("Cluster (ước tính): **Rủi ro Trung bình (Medium Risk)** – Cần xem xét thêm")
        else:
            st.error("Cluster (ước tính): **Rủi ro Cao (High Risk)** – Nguy cơ vỡ nợ cao")

else:
    st.info("👈 Điền thông tin vào thanh bên trái rồi nhấn nút phê duyệt.\n\n"
            "*(Fill in the left sidebar and click the approve button.)*")
