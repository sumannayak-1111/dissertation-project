"""Respiratory disease detection Streamlit application.
I load the artefacts persisted by final_code.ipynb (three base models, the tuned
final model, the scaler, the label encoder, the feature column ordering, and a
metadata dictionary) and expose them through a clean clinician-facing interface.
The user enters symptom information and basic demographics, picks which model
to run, and receives the top three predicted respiratory conditions with a
SHAP-based explanation panel and a clear medical disclaimer.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).parent / "models"
SYMPTOM_GROUPS = {
    "Lower respiratory symptoms": [
        ("cough_any", "Cough (any type)"),
        ("dry_cough", "Dry cough"),
        ("productive_cough", "Productive cough (sputum)"),
        ("chronic_cough", "Chronic cough (longer than 3 weeks)"),
        ("wheezing", "Wheezing"),
        ("shortness_of_breath", "Shortness of breath"),
        ("chest_tightness", "Chest tightness"),
        ("pleuritic_chest_pain", "Pleuritic chest pain"),
        ("tachypnea", "Rapid breathing (tachypnea)"),
        ("hemoptysis", "Coughing up blood (hemoptysis)"),
    ],
    "Upper respiratory symptoms": [
        ("sore_throat", "Sore throat"),
        ("runny_nose", "Runny nose"),
        ("nasal_congestion", "Nasal congestion"),
        ("sneezing", "Sneezing"),
        ("loss_of_smell_taste", "Loss of smell or taste"),
    ],
    "Constitutional symptoms": [
        ("fever", "Fever"),
        ("high_fever", "High fever (above 38.5 C)"),
        ("night_sweats", "Night sweats"),
        ("chills", "Chills or rigors"),
        ("fatigue", "Fatigue"),
        ("weight_loss", "Unintentional weight loss"),
        ("body_aches", "Body aches or myalgia"),
        ("headache", "Headache"),
    ],
    "Functional impact": [
        ("sleep_disturbance", "Sleep disturbance"),
        ("activity_limitation", "Activity limitation"),
    ],
}
ALL_SYMPTOM_KEYS = [key for group in SYMPTOM_GROUPS.values() for key, _ in group]
MODEL_FILES: dict[str, str] = {
    "Final tuned model (recommended)": "final_tuned_model.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "XGBoost": "xgboost.joblib",
    "MLP": "mlp.joblib",
}
@st.cache_resource(show_spinner=False)
def load_artefacts() -> dict:
    """Load every persisted artefact from the models directory."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"The models directory was not found at {MODELS_DIR}. "
            "Run final_code.ipynb top to bottom first to produce the saved artefacts."
        )
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_columns: list[str] = joblib.load(MODELS_DIR / "feature_columns.joblib")
    metadata: dict = joblib.load(MODELS_DIR / "model_metadata.joblib")
    models = {}
    for display_name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[display_name] = joblib.load(path)
    return {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "models": models,
    }
def build_input_row(
    feature_columns: list[str],
    symptom_values: dict[str, int],
    age: int,
    sex: str,
    smoking_status: str,
    occupational_exposure: int,
    spo2: float,
    respiratory_rate: int,
    symptom_duration_days: int,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the exact column order the model expects."""
    row: dict[str, float] = {col: 0 for col in feature_columns}
    for symptom in ALL_SYMPTOM_KEYS:
        if symptom in row:
            row[symptom] = int(symptom_values.get(symptom, 0))
    if "age" in row:
        row["age"] = age
    if "occupational_exposure" in row:
        row["occupational_exposure"] = int(occupational_exposure)
    if "spo2" in row:
        row["spo2"] = float(spo2)
    if "respiratory_rate" in row:
        row["respiratory_rate"] = int(respiratory_rate)
    if "symptom_duration_days" in row:
        row["symptom_duration_days"] = int(symptom_duration_days)
    if "sex_male" in row:
        row["sex_male"] = 1 if sex == "male" else 0
    for key in ["smoking_status_current", "smoking_status_former", "smoking_status_never"]:
        if key in row:
            row[key] = 0
    smoking_col = f"smoking_status_{smoking_status}"
    if smoking_col in row:
        row[smoking_col] = 1
    return pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)
def scale_continuous(df: pd.DataFrame, scaler, continuous_features: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    present = [c for c in continuous_features if c in scaled.columns]
    if present:
        scaled[present] = scaler.transform(scaled[present])
    return scaled
def predict_top_k(model, X: pd.DataFrame, label_encoder, k: int = 3):
    probas = model.predict_proba(X)[0]
    order = np.argsort(probas)[::-1][:k]
    classes = label_encoder.classes_
    return [(classes[i], float(probas[i])) for i in order]
def compute_shap_contribution(
    model,
    X_scaled: pd.DataFrame,
    feature_columns: list[str],
    label_encoder,
    background: pd.DataFrame,
    target_class: str,
) -> pd.DataFrame:
    """Return a DataFrame of feature contributions for the target class."""
    target_idx = list(label_encoder.classes_).index(target_class)
    if isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_scaled)
        if isinstance(raw, list):
            shap_for_class = raw[target_idx][0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            shap_for_class = raw[0, :, target_idx]
        else:
            shap_for_class = raw[0]
    elif isinstance(model, LogisticRegression):
        coefficients = model.coef_[target_idx]
        shap_for_class = coefficients * X_scaled.iloc[0].values
    else:
        bg = background.iloc[:60]
        predict_fn: Callable = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_columns))
        explainer = shap.KernelExplainer(predict_fn, bg)
        raw = explainer.shap_values(X_scaled, nsamples=80, silent=True)
        if isinstance(raw, list):
            shap_for_class = raw[target_idx][0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            shap_for_class = raw[0, :, target_idx]
        else:
            shap_for_class = raw[0]
    contrib = pd.DataFrame({
        "feature": feature_columns,
        "shap": np.asarray(shap_for_class).flatten(),
        "value": X_scaled.iloc[0].values,
    })
    contrib["abs"] = contrib["shap"].abs()
    return contrib.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
def render_probability_chart(predictions: list[tuple[str, float]]) -> None:
    """Render a horizontal bar chart of the top-k predicted probabilities."""
    diseases = [name for name, _ in predictions][::-1]
    probs = [p * 100.0 for _, p in predictions][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(diseases, probs, color=["#2a9d8f", "#e9c46a", "#f4a261"][: len(diseases)][::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Predicted probability (%)")
    ax.set_title("Top predicted respiratory conditions")
    for bar, p in zip(bars, probs):
        ax.text(min(p + 1.5, 95), bar.get_y() + bar.get_height() / 2, f"{p:.1f}%",
                va="center", fontsize=10)
    st.pyplot(fig, clear_figure=True)
def render_shap_panel(contrib: pd.DataFrame, predicted_class: str, top_n: int = 12) -> None:
    top = contrib.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c9be8" if v >= 0 else "#e07a5f" for v in top["shap"]]
    ax.barh(top["feature"], top["shap"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Contribution toward this prediction")
    ax.set_title(f"Top feature contributions for predicted class: {predicted_class}")
    st.pyplot(fig, clear_figure=True)
def render_disclaimer() -> None:
    st.warning(
        "This application is a clinical decision support prototype built on synthetic "
        "data for a postgraduate dissertation. It is not a medical device and must not "
        "be used to diagnose, treat, or rule out disease. Always consult a qualified "
        "clinician for medical advice."
    )
def main() -> None:
    st.set_page_config(
        page_title="Respiratory Disease Detection Tool",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Respiratory Disease Detection Tool")
    st.caption("Machine learning powered clinical decision support across seven respiratory conditions")
    render_disclaimer()
    try:
        artefacts = load_artefacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    scaler = artefacts["scaler"]
    label_encoder = artefacts["label_encoder"]
    feature_columns: list[str] = artefacts["feature_columns"]
    metadata: dict = artefacts["metadata"]
    models = artefacts["models"]
    continuous_features = metadata.get(
        "continuous_features",
        ["age", "spo2", "respiratory_rate", "symptom_duration_days"],
    )
    with st.sidebar:
        st.header("Model and patient information")
        available_model_names = [name for name in MODEL_FILES.keys() if name in models]
        selected_model_name = st.selectbox(
            "Choose the prediction model",
            available_model_names,
            index=0,
        )
        st.subheader("Demographics")
        age = st.slider("Age (years)", min_value=0, max_value=100, value=45)
        sex = st.radio("Sex", options=["male", "female"], index=0, horizontal=True)
        smoking_status = st.radio(
            "Smoking status",
            options=["never", "former", "current"],
            index=0,
            horizontal=True,
        )
        occupational_exposure = st.checkbox(
            "Occupational exposure to dust, fumes or irritants", value=False
        )
        st.subheader("Clinical observations")
        spo2 = st.slider("Oxygen saturation SpO2 (%)", min_value=70.0, max_value=100.0,
                         value=97.0, step=0.5)
        respiratory_rate = st.slider("Respiratory rate (breaths per minute)",
                                     min_value=8, max_value=50, value=18)
        symptom_duration_days = st.slider("Symptom duration (days)",
                                          min_value=1, max_value=365, value=5)
        st.markdown("---")
        st.caption(
            f"Disease classes: {', '.join(metadata.get('disease_classes', list(label_encoder.classes_)))}"
        )
        st.caption(
            f"Tuned winner: {metadata.get('winner', 'unknown')} | random seed: "
            f"{metadata.get('random_state', 42)}"
        )
    st.subheader("Symptom checklist")
    st.caption("Tick every symptom the patient reports.")
    symptom_values: dict[str, int] = {}
    for group_name, group_items in SYMPTOM_GROUPS.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(2)
            for i, (key, label) in enumerate(group_items):
                with cols[i % 2]:
                    symptom_values[key] = int(st.checkbox(label, key=f"chk_{key}"))
    predict_button = st.button("Predict respiratory condition", type="primary", use_container_width=True)
    if predict_button:
        if sum(symptom_values.values()) == 0:
            st.warning("No symptoms were ticked. The prediction will reflect baseline class priors.")
        input_row = build_input_row(
            feature_columns=feature_columns,
            symptom_values=symptom_values,
            age=age,
            sex=sex,
            smoking_status=smoking_status,
            occupational_exposure=int(occupational_exposure),
            spo2=spo2,
            respiratory_rate=respiratory_rate,
            symptom_duration_days=symptom_duration_days,
        )
        input_scaled = scale_continuous(input_row, scaler, continuous_features)
        model = models[selected_model_name]
        with st.spinner("Running model inference"):
            top_predictions = predict_top_k(model, input_scaled, label_encoder, k=3)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Top 3 predictions")
            for rank, (name, prob) in enumerate(top_predictions, start=1):
                st.markdown(f"**{rank}. {name}**")
                st.progress(min(int(prob * 100), 100))
                st.caption(f"Probability: {prob * 100:.2f}%")
            render_probability_chart(top_predictions)
        with right:
            st.subheader("Why this prediction")
            predicted_class = top_predictions[0][0]
            try:
                with st.spinner("Computing SHAP-based explanation"):
                    contrib = compute_shap_contribution(
                        model=model,
                        X_scaled=input_scaled,
                        feature_columns=feature_columns,
                        label_encoder=label_encoder,
                        background=scale_continuous(
                            pd.DataFrame([[0] * len(feature_columns)] * 80, columns=feature_columns),
                            scaler,
                            continuous_features,
                        ),
                        target_class=predicted_class,
                    )
                render_shap_panel(contrib, predicted_class)
                st.caption(
                    "Positive contributions push the prediction toward the top class; "
                    "negative contributions push against it."
                )
            except Exception as exc:
                st.info(f"Explanation panel could not be generated for this model: {exc}")
        st.markdown("---")
        with st.expander("Model performance reference"):
            metrics = metadata.get("test_metrics", {})
            if metrics:
                rows = []
                for key, vals in metrics.items():
                    pretty = key.replace("_", " ").title()
                    rows.append({
                        "Model": pretty,
                        "Accuracy": f"{vals.get('accuracy', float('nan')):.4f}" if "accuracy" in vals else "n/a",
                        "Macro-F1": f"{vals.get('f1_macro', float('nan')):.4f}",
                        "Weighted-F1": f"{vals.get('f1_weighted', float('nan')):.4f}" if "f1_weighted" in vals else "n/a",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("---")
    st.caption(
        "Project: Machine Learning Powered Respiratory Disease Detection Tool. "
        "Built for postgraduate dissertation work. "
        "Outputs are derived from a clinically grounded synthetic dataset and are "
        "intended for academic demonstration only."
    )
if __name__ == "__main__":
    main()
