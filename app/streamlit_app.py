import streamlit as st
import tempfile
from src.infer import predict

st.title("Music Genre Classification Demo")

uploaded = st.file_uploader("Upload an audio file (mp3/wav)", type=["mp3", "wav"])
if uploaded is not None:
    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.audio(uploaded)

    try:
        top5 = predict(tmp_path, topk=5)
        st.subheader("Top-5 genres")
        for label, p in top5:
            st.write(f"- **{label}**: {p:.3f}")
    except Exception as e:
        st.error(str(e))
        st.info("Tip: Train model first (python -m src.preprocess then python -m src.train).")
