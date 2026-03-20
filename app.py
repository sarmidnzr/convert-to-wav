import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile, os
import nnaud as n

st.set_page_config(page_title="Audio Digit Classifier", page_icon="🎙️")
st.title("Audio Digit Classifier")

LABELS = {0:"zero",1:"one",2:"two",3:"three",4:"four",
          5:"five",6:"six",7:"seven",8:"eight",9:"nine"}

@st.cache_resource
def load_net():
    return n.Neural_Net()

net = load_net()

def predict_raw(raw: np.ndarray) -> str:
    feats = n.app_extract_normalized_features(live_audio=raw)
    digit = net.app_inference(feats)[1]
    return LABELS[digit]

def predict_file(path: str) -> str:
    feats = n.app_extract_normalized_features(file_path=path)
    digit = net.app_inference(feats)[1]
    return LABELS[digit]

tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎙️ Microphone"])

with tab_upload:
    f = st.file_uploader("Upload a WAV file of a spoken digit (0–9)", type=["wav"])
    if f and st.button("Classify", key="cls_file"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        with st.spinner("Running inference…"):
            result = predict_file(tmp_path)
        os.unlink(tmp_path)
        st.success(f"Predicted: {result.upper()}")

with tab_mic:
    duration = st.slider("Recording duration (seconds)", 1, 4, 2)
    if st.button("⏺ Record & Classify", key="cls_mic"):
        with st.spinner(f"Recording for {duration}s…"):
            recording = sd.rec(int(duration * 16000), samplerate=16000,
                               channels=1, dtype="float32")
            sd.wait()
        with st.spinner("Running inference…"):
            result = predict_raw(recording[:, 0])
        st.success(f"Predicted: {result.upper()}")