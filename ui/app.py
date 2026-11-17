"""Streamlit frontend for the chest X-ray pneumonia classifier.

Run locally (CMD), with the API already running on port 8000:
    .venv\\Scripts\\activate
    streamlit run ui/app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SAMPLES_DIR = Path(__file__).parent / "samples"
TIMEOUT_S = 60

DISCLAIMER_HTML = """
<div style="background:#b00020;color:#fff;padding:18px 22px;border-radius:10px;
            border:3px solid #7a0016;margin-bottom:18px;font-size:1.05rem;line-height:1.45">
  <div style="font-size:1.35rem;font-weight:800;letter-spacing:.3px;margin-bottom:6px">
    ⚠️ EDUCATIONAL DEMO ONLY: NOT A MEDICAL DEVICE
  </div>
  This tool is <b>not for clinical use</b>. It must not be used to diagnose, screen,
  or make any decision about a real patient. It was trained on one pediatric dataset
  from one hospital, has <b>not been clinically validated</b>, and produces frequent
  false alarms (test specificity ~0.74). Always consult a qualified clinician.
</div>
"""

ATTRIBUTION = (
    "Sample images: Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) "
    "and Chest X-Ray Images for Classification*, Mendeley Data v2, "
    "[doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2), licensed "
    "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Downscaled and re-encoded; "
    "randomly selected from the held-out test set."
)

st.set_page_config(page_title="Chest X-ray Pneumonia Classifier (demo)", page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------- API helpers
@st.cache_data(ttl=60, show_spinner=False)
def get_json(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(data: bytes, filename: str) -> dict:
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    try:
        r = requests.post(f"{API_URL}/predict", files={"file": (filename, data, mime)}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return {"error": f"Could not reach the API at {API_URL}: {e}"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return {"error": f"API error {r.status_code}: {detail}"}
    return r.json()


def fmt_prob(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else ("> 0.9999" if p > 0.9999 else f"{p:.4f}")


def probability_bar(p: float, thr: float) -> str:
    color = "#c62828" if p >= thr else "#2e7d32"
    return f"""
    <div style="margin:6px 0 2px 0;font-size:.9rem">Pneumonia probability</div>
    <div style="position:relative;height:26px;background:#e6e6e6;border-radius:6px;overflow:visible">
      <div style="width:{p * 100:.2f}%;height:100%;background:{color};border-radius:6px"></div>
      <div title="threshold" style="position:absolute;left:{thr * 100:.2f}%;top:-6px;bottom:-6px;
           width:3px;background:#111"></div>
    </div>
    <div style="position:relative;height:20px;font-size:.8rem;color:#555">
      <span style="position:absolute;left:0">0</span>
      <span style="position:absolute;left:calc({thr * 100:.2f}% - 40px)">threshold {thr:.3f}</span>
      <span style="position:absolute;right:0">1</span>
    </div>
    """


# --------------------------------------------------------------------------- layout
st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
st.title("🫁 Chest X-ray Pneumonia Classifier")
st.caption(
    "DenseNet121 transfer learning · NORMAL vs PNEUMONIA · Grad-CAM explanations · "
    "FastAPI backend · educational portfolio project"
)

# Sidebar: model card
with st.sidebar:
    st.header("Model card")
    health = get_json("/health")
    if health and health.get("model_loaded"):
        st.success("API online · model loaded")
    elif health:
        st.warning(f"API online, model NOT loaded: {health.get('error')}")
    else:
        st.error(f"API unreachable at {API_URL}")

    card = get_json("/model-info")
    if card:
        tm = card.get("test_metrics", {})
        c1, c2 = st.columns(2)
        c1.metric("Test ROC-AUC", f"{tm.get('auc', float('nan')):.3f}")
        c2.metric("Threshold", f"{card.get('threshold', float('nan')):.3f}")
        c1.metric("Sensitivity", f"{tm.get('sensitivity', float('nan')):.3f}")
        c2.metric("Specificity", f"{tm.get('specificity', float('nan')):.3f}")
        st.caption(
            f"Held-out test set (n={tm.get('tp', 0) + tm.get('fn', 0) + tm.get('tn', 0) + tm.get('fp', 0)}), "
            "evaluated once. Threshold chosen on the validation set to target ≥ 0.95 sensitivity."
        )
        st.warning(
            f"**Weakness:** specificity {tm.get('specificity', 0):.2f} means about "
            f"{tm.get('fp', '?')} of {tm.get('tn', 0) + tm.get('fp', 0)} normal test X-rays "
            "were wrongly flagged as pneumonia."
        )
        with st.expander("Details"):
            st.markdown(
                f"- **Architecture:** {card.get('architecture')} ({card.get('pretrained_on', '')})\n"
                f"- **Data:** {card.get('dataset', {}).get('population', 'n/a')}\n"
                f"- **License:** {card.get('dataset', {}).get('license', 'n/a')}\n"
                f"- **Created:** {card.get('created_utc', 'n/a')}"
            )
            st.json(card.get("pre_registered_criteria", {}))
    else:
        st.info("Model card unavailable.")
    st.divider()
    st.caption(f"API: `{API_URL}`")

# Input
st.subheader("1 · Choose an X-ray")
tab_sample, tab_upload = st.tabs(["Try a sample", "Upload your own"])

if "selected" not in st.session_state:
    st.session_state.selected = None  # (bytes, name, true_label | None)

with tab_sample:
    manifest = json.loads((SAMPLES_DIR / "samples.json").read_text())
    cols = st.columns(len(manifest))
    for col, s in zip(cols, manifest, strict=True):
        path = SAMPLES_DIR / s["file"]
        with col:
            st.image(str(path), width="stretch")
            if st.button(f"Use {s['file'].split('.')[0].replace('_', ' ')}", key=s["file"]):
                st.session_state.selected = (path.read_bytes(), s["file"], s["label"])
            st.caption(f"Ground truth: **{s['label']}**")
    st.caption(ATTRIBUTION)

with tab_upload:
    up = st.file_uploader("PNG or JPEG chest X-ray (max 10 MB)", type=["png", "jpg", "jpeg"])
    if up is not None and st.button("Analyze uploaded image", type="primary"):
        st.session_state.selected = (up.getvalue(), up.name, None)

# Result
sel = st.session_state.selected
if sel:
    data, name, truth = sel
    st.subheader("2 · Result")
    with st.spinner("Running model and Grad-CAM…"):
        res = predict(data, name)
    if "error" in res:
        st.error(res["error"])
    else:
        p, thr, label = res["pneumonia_probability"], res["threshold"], res["label"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", label)
        m2.metric("Pneumonia probability", fmt_prob(p))
        m3.metric("Threshold", f"{thr:.3f}")
        m4.metric("Inference", f"{res['inference_ms']:.0f} ms")
        st.markdown(probability_bar(p, thr), unsafe_allow_html=True)
        if truth:
            ok = truth == label
            (st.success if ok else st.error)(
                f"Ground truth for this sample: **{truth}**. The model was {'correct' if ok else 'WRONG'}."
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**Original X-ray**")
            st.image(Image.open(io.BytesIO(data)), width="stretch")
        with right:
            st.markdown(f"**Grad-CAM: regions driving the *{res['gradcam_class']}* decision**")
            st.image(base64.b64decode(res["gradcam_png_base64"]), width="stretch")
        st.caption(
            "Grad-CAM highlights image regions that most increased the score for the predicted class "
            "(red = strongest). It is coarse (7×7 grid upsampled), is not a lesion segmentation, and "
            "can highlight non-anatomical artifacts (text markers, borders). "
            "It does not show that the model reasons like a radiologist."
        )
        if truth:
            st.caption(ATTRIBUTION)

st.divider()
st.caption(
    "⚠️ Educational demo, not a medical device, not for clinical use. "
    "Dataset: Kermany et al. (2018), CC BY 4.0."
)
