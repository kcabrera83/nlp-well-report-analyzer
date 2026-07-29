import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="NLP Well Report Analyzer", layout="wide")
st.title("NLP Well Report Analyzer")
st.markdown("Analyze well reports and extract structured information using NLP.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("classifier", "report_classifier.pkl"), ("sentiment", "sentiment_model.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
report_text = st.sidebar.text_area("Report Text", height=100)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[report_text]])
        m = models["classifier"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Classifier", result if isinstance(result, str) else f"{result:.4f}")
        m = models["sentiment"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Sentiment", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

