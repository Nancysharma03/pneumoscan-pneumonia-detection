import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="PneumoScan | X-ray AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMAGE_SIZE = (224, 224)


# Download the public model from Hugging Face and cache it for this app session.
@st.cache_resource(show_spinner="Downloading and loading the trained CNN model…")
def load_pneumonia_model():
    import tensorflow as tf

    model_path = hf_hub_download(
        repo_id="Nancy1303/pneumoscan-cnn-model",
        filename="best_cnn_model.keras",
    )
    return tf.keras.models.load_model(model_path, compile=False)


# High-contrast, readable palette. Streamlit's default theme can otherwise
# make custom HTML text hard to read when the user's app theme differs.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { color-scheme: light; }
    .stApp { background: #F4F7FB !important; color: #172B4D !important; }
    [data-testid="stHeader"] { background: #F4F7FB !important; }
    [data-testid="stAppViewContainer"] { background: #F4F7FB !important; }
    [data-testid="stSidebar"] { background: #E8EEF5 !important; border-right: 1px solid #D5DFEA; }
    [data-testid="stSidebar"] * { color: #203650 !important; }
    html, body, [class*="css"], p, label, span, li { font-family: 'DM Sans', sans-serif; }
    p, label, li, .stMarkdown { color: #263B53; }
    h1, h2, h3, h4 { color: #132A43 !important; font-family: 'Manrope', sans-serif !important; }
    [data-testid="stFileUploader"] { background: #FFFFFF; border: 1px dashed #91A7BD; border-radius: 14px; padding: 12px; }
    [data-testid="stFileUploader"] * { color: #203650 !important; }
    [data-testid="stAlert"] { color: #203650 !important; }
    [data-testid="stAlert"] p { color: #203650 !important; }
    [data-testid="stImage"] img { border-radius: 14px; border: 1px solid #DCE5EE; }
    .hero { background: linear-gradient(120deg,#102A43 0%,#174A63 58%,#087E8B 100%);
      padding: 34px 38px; border-radius: 22px; color: #FFFFFF; margin-bottom: 24px;
      box-shadow: 0 12px 28px rgba(16,42,67,.13); }
    .hero .eyebrow { color: #A8F0E8 !important; text-transform: uppercase; letter-spacing: .15em;
      font-size: .72rem; font-weight: 800; margin-bottom: 12px; }
    .hero h1 { color: #FFFFFF !important; font-size: clamp(2rem,3.2vw,2.8rem); line-height: 1.15; margin: 0 0 12px 0; }
    .hero p { color: #E4F0F7 !important; max-width: 760px; font-size: 1rem; line-height: 1.75; margin: 0; }
    .section-title { font-family: 'Manrope', sans-serif; color: #132A43 !important; font-size: 1.28rem; font-weight: 800; }
    .subtext { color: #52677D !important; font-size: .93rem; line-height: 1.6; }
    .panel { background: #FFFFFF; border: 1px solid #DCE5EE; border-radius: 16px; padding: 20px 22px; }
    .metric { background: #FFFFFF; border: 1px solid #DCE5EE; border-radius: 15px; padding: 18px 20px; }
    .metric-label { color: #52677D !important; font-size: .82rem; margin-bottom: 7px; }
    .metric-value { color: #132A43 !important; font-family: 'Manrope',sans-serif; font-size: 1.6rem; font-weight: 800; }
    .metric-note { color: #65798D !important; font-size: .78rem; margin-top: 4px; }
    .result { border-radius: 16px; padding: 22px; border: 1px solid; margin: 10px 0 16px 0; }
    .result-normal { background: #E9F8F1; border-color: #8DD8B6; }
    .result-pneumonia { background: #FFF3E6; border-color: #F2BE82; }
    .result-error { background: #FDECEC; border-color: #F4AAAA; }
    .result-kicker { font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
    .result-title { font-family: 'Manrope',sans-serif; font-size: 1.45rem; font-weight: 800; line-height: 1.3; margin-bottom: 8px; }
    .result-body { font-size: .94rem; line-height: 1.6; }
    .normal-text { color: #126344 !important; }
    .pneumonia-text { color: #8A4600 !important; }
    .error-text { color: #8F2020 !important; }
    .disclaimer { background: #FFF7E8; border: 1px solid #F3D39A; border-left: 5px solid #D97706;
      border-radius: 12px; padding: 15px 17px; color: #713F12 !important; line-height: 1.6; font-size: .88rem; }
    .disclaimer strong { color: #713F12 !important; }
    div.stButton > button[kind="primary"] { background: #087E8B !important; color: white !important;
      border: 1px solid #087E8B !important; border-radius: 10px; min-height: 3rem; font-weight: 800; }
    div.stButton > button[kind="primary"]:hover { background: #066875 !important; border-color: #066875 !important; }
    div.stButton > button:not([kind="primary"]) { background: #FFFFFF !important; color: #203650 !important; border: 1px solid #C9D5E2 !important; }
    [data-testid="stProgressBar"] > div { background: #D9E5ED !important; }
    [data-testid="stProgressBar"] [data-testid="stProgressBarFilled"] { background: #087E8B !important; }
    code { color: #075985 !important; background: #EAF4F8 !important; }
    hr { border-color: #D8E2EC !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)


def get_prediction(model, image: Image.Image):
    score = float(np.asarray(model.predict(preprocess_image(image), verbose=0)).reshape(-1)[0])
    label = "PNEUMONIA" if score > 0.5 else "NORMAL"
    confidence = score if label == "PNEUMONIA" else 1 - score
    return label, confidence, score


with st.sidebar:
    st.markdown("## 🩻 PneumoScan")
    st.markdown('<p class="subtext">Chest X-ray classification demo</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["X-ray screening", "About the project"])
    st.divider()
    st.markdown("### Model details")
    st.markdown("- **Architecture:** CNN")
    st.markdown("- **Input:** 224 × 224 RGB")
    st.markdown("- **Classes:** Normal / Pneumonia")
    st.markdown(
        '<div class="disclaimer"><strong>Educational prototype only</strong><br>'
        'Not a medical device. Do not use for diagnosis or treatment decisions.</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-assisted medical imaging · Project demo</div>
      <h1>Chest X-ray screening workspace</h1>
      <p>Upload a chest X-ray to view your trained CNN's classification output. This interface is built for learning and project demonstration, not clinical decision-making.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "About the project":
    st.markdown('<div class="section-title">Project overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtext">The project classifies chest X-ray images into the dataset labels NORMAL and PNEUMONIA. '
        'Your notebook evaluates a CNN baseline and a ResNet50 transfer-learning model.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, label, value, note in [
        (c1, "Training images", "5,216", "Before the 80/20 split"),
        (c2, "Held-out test images", "624", "Notebook test split"),
        (c3, "CNN test accuracy", "90.87%", "Reported notebook result"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("### How it works")
    st.markdown(
        """
        <div class="panel">
        <ol>
          <li>Upload a JPG, JPEG, or PNG image.</li>
          <li>The app converts it to RGB, resizes it to 224 × 224, and scales pixels to 0–1.</li>
          <li>The saved CNN returns a sigmoid score for the PNEUMONIA class.</li>
          <li>A 0.5 threshold maps the score to NORMAL or PNEUMONIA.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "The reported test accuracy comes from the supplied notebook. "
        "It does not guarantee performance on uploaded images or images from other settings."
    )
    st.stop()


st.markdown('<div class="section-title">Analyze an image</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtext">Choose a clear frontal chest X-ray image. Supported formats: JPG, JPEG, PNG.</p>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])


if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("### Uploaded X-ray")
            st.image(image, use_container_width=True)
            st.markdown(
                f'<p class="subtext">File: {uploaded_file.name}<br>'
                f'Dimensions: {image.width} × {image.height} px</p>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("### AI analysis")

            if st.button("Run X-ray analysis", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running the trained CNN…"):
                        model = load_pneumonia_model()
                        label, confidence, pneumonia_score = get_prediction(model, image)

                    st.session_state["prediction"] = {
                        "filename": uploaded_file.name,
                        "label": label,
                        "confidence": confidence,
                        "score": pneumonia_score,
                    }
                    st.session_state.pop("prediction_error", None)

                except Exception as exc:
                    st.session_state["prediction_error"] = str(exc)

            result = st.session_state.get("prediction")

            if st.session_state.get("prediction_error"):
                st.error("The model could not analyze this image. Check the deployment logs.")
                st.code(st.session_state["prediction_error"])
                st.session_state.pop("prediction_error", None)

            elif result and result["filename"] == uploaded_file.name:
                if result["label"] == "PNEUMONIA":
                    st.markdown(
                        f'<div class="result result-pneumonia">'
                        f'<div class="result-kicker pneumonia-text">Model prediction</div>'
                        f'<div class="result-title pneumonia-text">Pneumonia class predicted</div>'
                        f'<div class="result-body pneumonia-text">Confidence for this predicted class: '
                        f'<b>{result["confidence"]:.1%}</b></div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result result-normal">'
                        f'<div class="result-kicker normal-text">Model prediction</div>'
                        f'<div class="result-title normal-text">Normal class predicted</div>'
                        f'<div class="result-body normal-text">Confidence for this predicted class: '
                        f'<b>{result["confidence"]:.1%}</b></div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("**Pneumonia class score**")
                st.progress(float(np.clip(result["score"], 0, 1)))
                st.markdown(
                    f'<p class="subtext">Raw score: <b>{result["score"]:.3f}</b> · '
                    f'Classification threshold: <b>0.500</b></p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="disclaimer"><strong>Important safety note</strong><br>'
                    'This is a model output, not a diagnosis. A low pneumonia score does not rule out illness. '
                    'Please consult a qualified healthcare professional for medical interpretation.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="panel"><b>Ready to analyze</b>'
                    '<p class="subtext">Click the button above to run the saved CNN model. '
                    'The classification and score will appear here.</p></div>',
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"Could not open this image: {exc}")

else:
    st.markdown(
        '<div class="panel"><div class="section-title">Ready when you are</div>'
        '<p class="subtext">Your uploaded image and model result will appear in two clear panels here.</p></div>',
        unsafe_allow_html=True,
    )


st.divider()
st.markdown(
    '<p class="subtext">PneumoScan · Student project prototype · CNN chest X-ray classification</p>',
    unsafe_allow_html=True,
)
