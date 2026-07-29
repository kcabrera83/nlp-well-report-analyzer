import streamlit as st, joblib, numpy as np
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Report Analyzer", page_icon="\U0001f4ca")
st.header("Report Analyzer")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'type': joblib.load(p / 'report_classifier.pkl'), 'sentiment': joblib.load(p / 'sentiment_model.pkl')}

with st.sidebar:
    length = st.slider('Length', 0, 10000, 5000)
    run = st.button('Analyze', use_container_width=True)

if run:
    x = np.array([[length]])
    st.divider()
    m = models['type']
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else f'{p[0]:.2f}'
    else:
        v = f'{m.predict(x)[0]:.2f}'
    st.metric('Type', v)
    m = models['sentiment']
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else f'{p[0]:.2f}'
    else:
        v = f'{m.predict(x)[0]:.2f}'
    st.metric('Sentiment', v)