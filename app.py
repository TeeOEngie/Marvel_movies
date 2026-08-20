# ============================================================
# app.py
# Marvel Box Office Predictor — Streamlit app
#
# Loads the Random Forest model trained in train_model.py and
# lets the user predict a movie's worldwide box office gross
# from budget, runtime, ratings, and MCU phase info.
#
# Repo layout expected:
#   app.py
#   requirements.txt
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
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Marvel Box Office Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS — Marvel-ish red / gold theme
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0d0d15 100%);
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.4rem;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #ed1d24, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .hero-subtitle {
        color: #b8b8c8;
        font-size: 1.05rem;
        margin-top: -8px;
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
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .prediction-label {
        color: #ffe4b5;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
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
# Data / model loading (cached so it only runs once per session)
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
    '<p class="hero-subtitle">An Ensemble Random Forest Regressor trained on the MCU box office dataset</p>',
    unsafe_allow_html=True,
)
st.write("")

if not model_loaded:
    st.error(
        "Model files not found. Make sure `model/rf_model.pkl`, "
        "`model/feature_columns.pkl`, and `model/metrics.json` "
        "(produced by `train_model.py`) are in the repo."
    )
    st.stop()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 Predict", "📊 Explore Dataset", "🧠 Model Details"]
)

# ============================================================
# TAB 1 — Predict
# ============================================================
with tab_predict:
    st.markdown(
        '<div class="caveat-box">⚠️ This model was trained on only 34 movies. '
        "Treat predictions as a fun estimate driven by historical MCU patterns, "
        "not a reliable financial forecast — see the Model Details tab for honest metrics.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.subheader("Movie inputs")

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input(
                "Production budget ($M)", min_value=1.0, max_value=1000.0,
                value=200.0, step=5.0,
            )
            runtime = st.number_input(
                "Runtime (minutes)", min_value=60, max_value=240, value=130, step=1
            )
            phase_number = st.selectbox("MCU Phase", [1, 2, 3, 4, 5, 6], index=4)
            timeline_order = st.number_input(
                "Timeline order (release sequence #)", min_value=1, max_value=60,
                value=int(df["mcu_timeline_order"].max()) + 1, step=1,
            )

        with c2:
            release_year = st.number_input(
                "Release year", min_value=2008, max_value=2030, value=2026, step=1
            )
            imdb_rating = st.slider("IMDb rating", 0.0, 10.0, 7.0, 0.1)
            rotten_tomatoes = st.slider("Rotten Tomatoes score (%)", 0, 100, 75, 1)
            metacritic_score = st.slider("Metacritic score", 0, 100, 65, 1)

        predict_clicked = st.button("🎬 Predict Box Office")

    with col_result:
        st.subheader("Prediction")
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

            # Uncertainty estimate: spread across the individual trees
            # in the forest (this is what makes it an *ensemble* —
            # every tree votes and we can see how much they disagree).
            tree_preds = np.array([t.predict(input_row)[0] for t in model.estimators_])
            lower, upper = np.percentile(tree_preds, [10, 90])

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted Worldwide Gross</div>
                    <div class="prediction-value">${prediction:,.0f}M</div>
                    <div class="prediction-label">80% range: ${lower:,.0f}M – ${upper:,.0f}M</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            roi_est = ((prediction - budget) / budget) * 100
            m1, m2 = st.columns(2)
            m1.metric("Est. Profit", f"${prediction - budget:,.0f}M")
            m2.metric("Est. ROI", f"{roi_est:,.0f}%")

            # Tree-vote distribution — shows the "ensemble" at work
            fig = px.histogram(
                tree_preds, nbins=20,
                labels={"value": "Predicted gross ($M)"},
                title="Prediction across all trees in the forest",
                color_discrete_sequence=["#ed1d24"],
            )
            fig.add_vline(x=prediction, line_color="#ffd700", line_width=3,
                           annotation_text="Ensemble average")
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill in the movie details and click **Predict Box Office**.")

# ============================================================
# TAB 2 — Explore Dataset
# ============================================================
with tab_explore:
    st.subheader("MCU Movies Dataset")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Movies", len(df))
    k2.metric("Avg. Budget", f"${df['production_budget_millions'].mean():,.0f}M")
    k3.metric("Avg. Worldwide Gross", f"${df['worldwide_box_office_millions'].mean():,.0f}M")
    k4.metric("Avg. IMDb Rating", f"{df['imdb_rating'].mean():.1f}")

    st.write("")
    left, right = st.columns(2)

    with left:
        fig1 = px.bar(
            df.sort_values("release_date"),
            x="movie_title", y="worldwide_box_office_millions",
            color="phase", title="Worldwide box office by movie",
        )
        fig1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-60, height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        fig2 = px.scatter(
            df, x="production_budget_millions", y="worldwide_box_office_millions",
            size="imdb_rating", color="phase", hover_name="movie_title",
            title="Budget vs. worldwide gross",
        )
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        df.sort_values("release_date"), x="release_date", y="imdb_rating",
        markers=True, title="IMDb rating over time",
    )
    fig3.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Raw data")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 — Model Details
# ============================================================
with tab_model:
    st.subheader("Ensemble Random Forest Regressor")
    st.markdown(
        "A Random Forest is itself an **ensemble** method: it trains many "
        "decision trees on bootstrapped samples of the data (bagging) and "
        "averages their individual predictions, which reduces overfitting "
        "compared to any single tree."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{metrics['test_r2']:.3f}")
    c2.metric("Test MAE", f"${metrics['test_mae_millions']:,.1f}M")
    c3.metric("Test RMSE", f"${metrics['test_rmse_millions']:,.1f}M")

    st.markdown(
        f"""
        <div class="caveat-box">
        With only <b>{metrics['n_movies']} movies</b> in the dataset, these metrics
        are noisy — 5-fold cross-validated R² averaged
        <b>{metrics['cv_r2_mean']:.2f} ± {metrics['cv_r2_std']:.2f}</b> across folds.
        This model is best treated as a demo of the ensemble-learning workflow,
        not a production forecasting tool.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Feature importance")
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

    st.subheader("Best hyperparameters (from GridSearchCV)")
    st.json(metrics["best_params"])

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Built with scikit-learn RandomForestRegressor + Streamlit · "
    "Trained in Google Colab on the Marvel Movies dataset."
)
