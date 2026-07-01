"""
Streamlit dashboard for the Energy Demand Forecasting project.

Run:
    streamlit run streamlit_app.py

Tabs
----
1. Model Info       — model name, R², training params
2. Predict          — upload scaled CSV → see real-unit forecast chart
3. Forecast         — enter last 24 values → autoregressive n-step chart
4. Metrics          — load metrics.json from model evaluation
"""

import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── must be first streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

# ── load pipeline (cached so it only loads once per session) ───────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    from src.dlProject_energy_demand_forcasting.pipeline.prediction_pipeline import PredictionPipeline
    return PredictionPipeline()

pipeline = load_pipeline()

from src.dlProject_energy_demand_forcasting.pipeline.prediction_pipeline import WINDOW_SIZE, TARGET_COLUMN

# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=64)
st.sidebar.title("⚡ Energy Forecasting")
st.sidebar.markdown(f"**Model:** `{pipeline.model_name}`")
st.sidebar.markdown(f"**R² (train):** `{pipeline.best_info.get('r2_score', 'N/A'):.4f}`")
st.sidebar.markdown(f"**Window size:** `{WINDOW_SIZE}` hours")
st.sidebar.markdown(f"**Target:** `{TARGET_COLUMN}`")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with TensorFlow · MLflow · Streamlit")

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Model Info", "📊 Predict", "🔮 Forecast", "📈 Metrics", "⚙️ Retrain"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Model Info
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model",  pipeline.model_name)
    col2.metric("R² Score",    f"{pipeline.best_info.get('r2_score', 0):.4f}")
    col3.metric("Window Size", f"{WINDOW_SIZE} hours")

    st.markdown("---")
    st.subheader("Training Parameters")

    params_path = os.path.join("params.yaml")
    if os.path.exists(params_path):
        import yaml
        with open(params_path) as f:
            params = yaml.safe_load(f)

        col_lstm, col_gru = st.columns(2)
        with col_lstm:
            st.markdown("**LSTM**")
            st.json(params.get("LSTM", {}))
        with col_gru:
            st.markdown("**GRU**")
            st.json(params.get("GRU", {}))
    else:
        st.info("params.yaml not found. Run from the project root directory.")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```
    Raw Data (.txt)
        ↓ DatetimeIndexer   (Date + Time → Datetime index)
        ↓ TimeSeriesImputer (time-based interpolation)
        ↓ TimeSeriesResampler (hourly averages)
        ↓ TimeSeriesScaling  (MinMaxScaler 0–1)
        ↓ create_sequences   (sliding window of 24 hours)
        ↓ LSTM / GRU model
        ↓ inverse_transform  → kW predictions
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Predict from CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Predict from Transformed CSV")
    st.markdown(
        f"Upload `test_transformed.csv` (or any transformed CSV with a `Datetime` index "
        f"and `{TARGET_COLUMN}` column). The model will predict on every window of {WINDOW_SIZE} hours."
    )

    uploaded = st.file_uploader("Upload transformed CSV", type=["csv"])

    # Fallback — use the artifact if it exists
    default_path = os.path.join("artifacts", "data_transformation", "test_transformed.csv")
    if uploaded is None and os.path.exists(default_path):
        st.info(f"No file uploaded — using `{default_path}`.")
        df_transformed = pd.read_csv(default_path, index_col="Datetime", parse_dates=True)
    elif uploaded is not None:
        df_transformed = pd.read_csv(uploaded, index_col="Datetime", parse_dates=True)
    else:
        df_transformed = None

    if df_transformed is not None:
        st.markdown(f"**Shape:** {df_transformed.shape} &nbsp;|&nbsp; **Columns:** {', '.join(df_transformed.columns)}")

        if TARGET_COLUMN not in df_transformed.columns:
            st.error(f"Column `{TARGET_COLUMN}` not found in uploaded file.")
        else:
            scaled_series = df_transformed[TARGET_COLUMN].values

            with st.spinner("Running predictions..."):
                result = pipeline.predict_from_scaled(scaled_series)

            preds   = result["predictions"]
            n_preds = len(preds)

            # Align timestamps
            timestamps = df_transformed.index[WINDOW_SIZE : WINDOW_SIZE + n_preds]
            actuals    = pipeline._inverse_transform(scaled_series[WINDOW_SIZE : WINDOW_SIZE + n_preds])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=actuals, name="Actual",    line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=timestamps, y=preds,   name="Predicted", line=dict(color="#ff7f0e", dash="dash")))
            fig.update_layout(
                title=f"{TARGET_COLUMN} — Actual vs Predicted (kW)",
                xaxis_title="Datetime",
                yaxis_title="Global Active Power (kW)",
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mae  = mean_absolute_error(actuals, preds)
            r2   = r2_score(actuals, preds)
            col_a.metric("RMSE", f"{rmse:.4f} kW")
            col_b.metric("MAE",  f"{mae:.4f} kW")
            col_c.metric("R²",   f"{r2:.4f}")

            with st.expander("Download predictions as CSV"):
                pred_df = pd.DataFrame({
                    "Datetime" : timestamps,
                    "Actual_kW": actuals,
                    "Predicted_kW": preds,
                })
                st.download_button(
                    "Download CSV",
                    pred_df.to_csv(index=False).encode(),
                    "predictions.csv",
                    "text/csv",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Future Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Autoregressive Future Forecast")
    st.markdown(
        f"Paste the last **{WINDOW_SIZE} scaled values** of `{TARGET_COLUMN}` "
        f"(from `test_transformed.csv`). The model will roll forward and predict future hours."
    )

    col_left, col_right = st.columns([2, 1])

    with col_right:
        n_steps = st.slider("Hours to forecast", min_value=1, max_value=168, value=24, step=1)
        use_tail = st.checkbox("Auto-fill from test_transformed.csv", value=True)

    with col_left:
        if use_tail and os.path.exists(default_path):
            df_tail = pd.read_csv(default_path, index_col="Datetime", parse_dates=True)
            last_window = df_tail[TARGET_COLUMN].values[-WINDOW_SIZE:].tolist()
            st.info(f"Using last {WINDOW_SIZE} values from `test_transformed.csv`.")
            st.line_chart(pd.Series(last_window, name="Last window (scaled)"))
        else:
            raw_input = st.text_area(
                f"Paste {WINDOW_SIZE} comma-separated scaled values:",
                value=", ".join(["0.30"] * WINDOW_SIZE),
                height=120,
            )
            try:
                last_window = [float(v.strip()) for v in raw_input.split(",")]
            except ValueError:
                st.error("Could not parse values. Ensure they are comma-separated floats.")
                last_window = []

    if st.button("🔮 Forecast", type="primary") and len(last_window) == WINDOW_SIZE:
        with st.spinner(f"Forecasting {n_steps} hours ahead..."):
            result = pipeline.predict_next_n(last_window=last_window, n_steps=n_steps)

        preds = result["predictions"]
        future_hours = list(range(1, n_steps + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_hours, y=preds,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e"),
            marker=dict(size=5),
        ))
        fig.update_layout(
            title=f"{n_steps}-Hour Autoregressive Forecast — {pipeline.model_name}",
            xaxis_title="Hours ahead",
            yaxis_title="Global Active Power (kW)",
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Peak (kW)",    f"{max(preds):.3f}")
        col_b.metric("Min (kW)",     f"{min(preds):.3f}")
        col_c.metric("Avg (kW)",     f"{np.mean(preds):.3f}")

        with st.expander("Download forecast"):
            fcast_df = pd.DataFrame({
                "Hour_Ahead"    : future_hours,
                "Forecast_kW"   : preds,
                "Forecast_scaled": result["predictions_scaled"],
            })
            st.download_button(
                "Download CSV",
                fcast_df.to_csv(index=False).encode(),
                "forecast.csv",
                "text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Model Evaluation Metrics")

    metrics_path = os.path.join("artifacts", "model_evaluation", "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        model_name = metrics.get("model_name", pipeline.model_name)
        st.markdown(f"### Results for: `{model_name}`")

        real_cols  = ["rmse",        "mae",        "r2_score",        "mape"]
        scale_cols = ["rmse_scaled", "mae_scaled", "r2_score_scaled", "mape_scaled"]
        labels     = ["RMSE",        "MAE",        "R²",              "MAPE (%)"]

        st.subheader("Real-unit metrics (kW)")
        cols = st.columns(4)
        for col, key, label in zip(cols, real_cols, labels):
            if key in metrics:
                col.metric(label, f"{metrics[key]:.4f}")

        if any(k in metrics for k in scale_cols):
            st.subheader("Scaled-space metrics (0–1)")
            cols2 = st.columns(4)
            for col, key, label in zip(cols2, scale_cols, labels):
                if key in metrics:
                    col.metric(label, f"{metrics[key]:.4f}")

        st.markdown("---")
        st.subheader("All metrics (raw JSON)")
        st.json(metrics)

        # Bar chart comparing real vs scaled RMSE/MAE
        if "rmse" in metrics and "rmse_scaled" in metrics:
            fig = go.Figure(data=[
                go.Bar(name="Real units (kW)", x=["RMSE", "MAE"],
                       y=[metrics.get("rmse", 0), metrics.get("mae", 0)],
                       marker_color="#1f77b4"),
                go.Bar(name="Scaled (0–1)", x=["RMSE", "MAE"],
                       y=[metrics.get("rmse_scaled", 0), metrics.get("mae_scaled", 0)],
                       marker_color="#ff7f0e"),
            ])
            fig.update_layout(
                barmode="group",
                title="RMSE & MAE — Real vs Scaled",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"`{metrics_path}` not found. "
            "Run the model evaluation pipeline first (`python main.py` or notebook 05)."
        )

with tab5: # Assuming you added the 5th tab
    st.header("⚙️ Model Retraining Control")
    st.warning("Retraining is resource-intensive and will take time. The dashboard will be unresponsive while training is active.")
    
    if st.button("🚀 Start Training Pipeline"):
        with st.spinner("Training in progress... This may take several minutes."):
            log_placeholder = st.empty()
            log_output = []
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            try:
                
                process = subprocess.Popen(
                    [sys.executable, "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  
                    encoding="utf-8",      
                    errors="replace",
                    env=my_env,
                    bufsize=1  # Line-buffered
                )
                for line in process.stdout:
                    log_output.append(line)
                    log_placeholder.code("".join(log_output[-30:]))
                
                process.wait()
                if process.returncode == 0:
                    st.success("Training completed successfully!")
                    st.rerun() 
                else:
                    st.error(f"Training failed with exit code {process.returncode}.")
                    st.text("Check the logs above for details.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")