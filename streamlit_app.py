import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="NLP Well Report Analyzer", layout="wide")
st.title("NLP Well Report Analyzer")
st.markdown("Analyze well reports using NLP.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'type': joblib.load(d / 'report_classifier.pkl'), 'sentiment': joblib.load(d / 'sentiment_model.pkl')}

st.sidebar.header("Input Parameters")
report_len = st.sidebar.slider('Report Len', 0, 10000, 5000)

if st.sidebar.button("Run"):
    try:
        x = np.array([[report_len]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))