# ============================================================
# app.py
# Marvel Box Office Predictor — แอป Streamlit (ภาษาไทย)
#
# โหลดโมเดล Random Forest ที่เทรนไว้จาก train_model.py แล้วให้ผู้ใช้
# ทำนายรายได้รวมทั่วโลก (worldwide box office) ของหนัง จากงบสร้าง,
# ความยาวหนัง, คะแนนรีวิว และข้อมูลเฟส MCU
#
# โครงสร้าง repo ที่ต้องมี:
#   app.py
#   requirements.txt
#   runtime.txt
#   marvel_movies_dataset.csv
#   model/
#     rf_model.pkl
#     feature_columns.pkl
#     metrics.json
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# ตั้งค่าหน้าเพจ (ต้องเป็นคำสั่ง Streamlit อันแรกสุด)
# ------------------------------------------------------------
st.set_page_config(
    page_title="เครื่องมือทำนายรายได้หนัง Marvel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS — ธีมแดง-ทอง สไตล์ Marvel
# ใช้ฟอนต์ Noto Sans Thai เพื่อให้ตัวอักษรไทยแสดงผลสวย
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+Thai:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Noto Sans Thai', 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0d0d15 100%);
    }

    .hero-title {
        font-family: 'Bebas Neue', 'Noto Sans Thai', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #ed1d24, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .hero-subtitle {
        color: #b8b8c8;
        font-size: 1.05rem;
        margin-top: -4px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f1f33 0%, #2a1520 100%);
        border: 1px solid #ed1d2440;
        border-radius: 14px;
        padding: 16px 18px;
    }

    div[data-testid="stMetricValue"] {
        color: #ffd700;
    }

    .prediction-card {
        background: linear-gradient(135deg, #ed1d24 0%, #8b0000 100%);
        border-radius: 18px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(237, 29, 36, 0.35);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .prediction-value {
        font-family: 'Bebas Neue', 'Noto Sans Thai', sans-serif;
        font-size: 2.8rem;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .prediction-label {
        color: #ffe4b5;
        font-size: 0.95rem;
        letter-spacing: 1px;
    }

    .stButton>button {
        background: linear-gradient(90deg, #ed1d24, #b8860b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        letter-spacing: 1px;
        width: 100%;
    }

    section[data-testid="stSidebar"] {
        background: #14141f;
        border-right: 1px solid #ed1d2430;
    }

    .caveat-box {
        background: #2a1f0d;
        border-left: 4px solid #ffd700;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #e8d9a8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# โหลดข้อมูล / โมเดล (cache ไว้ให้รันแค่ครั้งเดียวต่อ session)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/rf_model.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    metrics = json.loads(Path("model/metrics.json").read_text())
    return model, feature_columns, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv("marvel_movies_dataset.csv")


try:
    model, FEATURES, metrics = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

df = load_dataset()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<p class="hero-title">MARVEL BOX OFFICE PREDICTOR</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">ทำนายรายได้รวมทั่วโลกของหนัง MCU ด้วย Ensemble Random Forest Regressor</p>',
    unsafe_allow_html=True,
)
st.write("")

if not model_loaded:
    st.error(
        "ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามี `model/rf_model.pkl`, "
        "`model/feature_columns.pkl`, และ `model/metrics.json` "
        "(ที่สร้างจาก `train_model.py`) อยู่ใน repo แล้ว"
    )
    st.stop()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 ทำนาย", "📊 สำรวจข้อมูล", "🧠 รายละเอียดโมเดล"]
)

# ============================================================
# TAB 1 — ทำนาย
# ============================================================
with tab_predict:
    st.markdown(
        '<div class="caveat-box">⚠️ โมเดลนี้เทรนจากข้อมูลแค่ 34 เรื่องเท่านั้น '
        "ควรมองผลลัพธ์เป็นการประมาณแบบสนุกๆ ที่อิงจากรูปแบบของหนัง MCU ในอดีต "
        "ไม่ใช่การพยากรณ์ทางการเงินที่แม่นยำ — ดูตัวเลขจริงได้ที่แท็บ "
        "รายละเอียดโมเดล</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.subheader("ข้อมูลหนัง")

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input(
                "งบสร้าง (ล้านดอลลาร์)", min_value=1.0, max_value=1000.0,
                value=200.0, step=5.0,
            )
            runtime = st.number_input(
                "ความยาวหนัง (นาที)", min_value=60, max_value=240, value=130, step=1
            )
            phase_number = st.selectbox("เฟส MCU", [1, 2, 3, 4, 5, 6], index=4)
            timeline_order = st.number_input(
                "ลำดับการฉาย (timeline order)", min_value=1, max_value=60,
                value=int(df["mcu_timeline_order"].max()) + 1, step=1,
            )

        with c2:
            release_year = st.number_input(
                "ปีที่ฉาย", min_value=2008, max_value=2030, value=2026, step=1
            )
            imdb_rating = st.slider("คะแนน IMDb", 0.0, 10.0, 7.0, 0.1)
            rotten_tomatoes = st.slider("คะแนน Rotten Tomatoes (%)", 0, 100, 75, 1)
            metacritic_score = st.slider("คะแนน Metacritic", 0, 100, 65, 1)

        predict_clicked = st.button("🎬 ทำนายรายได้")

    with col_result:
        st.subheader("ผลการทำนาย")
        if predict_clicked:
            input_row = pd.DataFrame([{
                "production_budget_millions": budget,
                "runtime_minutes": runtime,
                "phase_number": phase_number,
                "mcu_timeline_order": timeline_order,
                "release_year": release_year,
                "imdb_rating": imdb_rating,
                "rotten_tomatoes": rotten_tomatoes,
                "metacritic_score": metacritic_score,
            }])[FEATURES]

            prediction = model.predict(input_row)[0]

            # ประมาณช่วงความไม่แน่นอน: ดูความกระจายของค่าที่แต่ละต้นไม้
            # ในป่าทำนายไว้ (นี่คือหัวใจของ "ensemble" — ทุกต้นไม้โหวต
            # แล้วเราดูได้ว่าแต่ละต้นเห็นต่างกันแค่ไหน)
            tree_preds = np.array([t.predict(input_row)[0] for t in model.estimators_])
            lower, upper = np.percentile(tree_preds, [10, 90])

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">รายได้รวมทั่วโลกที่คาดการณ์</div>
                    <div class="prediction-value">${prediction:,.0f}M</div>
                    <div class="prediction-label">ช่วงความเป็นไปได้ 80%: ${lower:,.0f}M – ${upper:,.0f}M</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            roi_est = ((prediction - budget) / budget) * 100
            m1, m2 = st.columns(2)
            m1.metric("กำไรโดยประมาณ", f"${prediction - budget:,.0f}M")
            m2.metric("ROI โดยประมาณ", f"{roi_est:,.0f}%")

            # กราฟกระจายผลโหวตของแต่ละต้นไม้ — แสดงให้เห็น "ensemble" ทำงานจริง
            fig = px.histogram(
                tree_preds, nbins=20,
                labels={"value": "รายได้ที่ทำนาย ($M)"},
                title="ผลการทำนายจากต้นไม้ทุกต้นในป่า (forest)",
                color_discrete_sequence=["#ed1d24"],
            )
            fig.add_vline(x=prediction, line_color="#ffd700", line_width=3,
                           annotation_text="ค่าเฉลี่ยของ Ensemble")
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("กรอกข้อมูลหนังทางซ้าย แล้วกด **ทำนายรายได้**")

# ============================================================
# TAB 2 — สำรวจข้อมูล
# ============================================================
with tab_explore:
    st.subheader("ชุดข้อมูลหนัง MCU")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("จำนวนหนัง", len(df))
    k2.metric("งบสร้างเฉลี่ย", f"${df['production_budget_millions'].mean():,.0f}M")
    k3.metric("รายได้รวมเฉลี่ย", f"${df['worldwide_box_office_millions'].mean():,.0f}M")
    k4.metric("คะแนน IMDb เฉลี่ย", f"{df['imdb_rating'].mean():.1f}")

    st.write("")
    left, right = st.columns(2)

    with left:
        fig1 = px.bar(
            df.sort_values("release_date"),
            x="movie_title", y="worldwide_box_office_millions",
            color="phase", title="รายได้รวมทั่วโลกแยกตามหนัง",
        )
        fig1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-60, height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        fig2 = px.scatter(
            df, x="production_budget_millions", y="worldwide_box_office_millions",
            size="imdb_rating", color="phase", hover_name="movie_title",
            title="งบสร้าง เทียบกับ รายได้รวมทั่วโลก",
        )
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        df.sort_values("release_date"), x="release_date", y="imdb_rating",
        markers=True, title="คะแนน IMDb ตามช่วงเวลา",
    )
    fig3.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("ข้อมูลดิบ")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 — รายละเอียดโมเดล
# ============================================================
with tab_model:
    st.subheader("Ensemble Random Forest Regressor")
    st.markdown(
        "Random Forest เป็นโมเดลแบบ **ensemble** ในตัวเองอยู่แล้ว: "
        "มันเทรน decision tree จำนวนมากจากตัวอย่างข้อมูลที่สุ่มมา (bagging) "
        "แล้วเฉลี่ยผลการทำนายของแต่ละต้น ซึ่งช่วยลดปัญหา overfitting "
        "เมื่อเทียบกับการใช้ต้นไม้แค่ต้นเดียว"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{metrics['test_r2']:.3f}")
    c2.metric("Test MAE", f"${metrics['test_mae_millions']:,.1f}M")
    c3.metric("Test RMSE", f"${metrics['test_rmse_millions']:,.1f}M")

    st.markdown(
        f"""
        <div class="caveat-box">
        เนื่องจากข้อมูลมีแค่ <b>{metrics['n_movies']} เรื่อง</b> ค่าตัวชี้วัดเหล่านี้
        จึงค่อนข้างแกว่ง — ค่า R² จากการทำ 5-fold cross-validation
        เฉลี่ยอยู่ที่ <b>{metrics['cv_r2_mean']:.2f} ± {metrics['cv_r2_std']:.2f}</b>
        ควรมองโมเดลนี้เป็นตัวอย่างสาธิตขั้นตอนการทำ ensemble learning
        มากกว่าเครื่องมือพยากรณ์ที่ใช้งานจริงได้แม่นยำ
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("ความสำคัญของแต่ละตัวแปร (Feature Importance)")
    importance_df = (
        pd.DataFrame(metrics["feature_importance"].items(), columns=["feature", "importance"])
        .sort_values("importance", ascending=True)
    )
    fig4 = px.bar(
        importance_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#ffd700", "#ed1d24"],
    )
    fig4.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", height=380, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("ค่า Hyperparameter ที่ดีที่สุด (จาก GridSearchCV)")
    st.json(metrics["best_params"])

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "สร้างด้วย scikit-learn RandomForestRegressor + Streamlit · "
    "เทรนโมเดลใน Google Colab จากชุดข้อมูล Marvel Movies"
)