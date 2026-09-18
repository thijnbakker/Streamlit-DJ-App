"""
DJ Genre Classifier — live demo
================================
Upload a track, get a genre prediction from the actual model trained for
my DJ Genre Classification project (see the full write-up on my portfolio).

This uses the real classifier.pkl (logistic regression + sigmoid calibration
on Essentia Discogs-EffNet embeddings), trained on ~2,600 hand-verified
labels across 38 genres, 76.7% cross-validated accuracy.

No audio is stored — everything happens in memory for the duration of the
request and is discarded afterward.
"""

import json
import os
import pickle
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────
HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
LABEL_ENC_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Essentia's pretrained feature extractor. ~17.5MB, downloaded once and
# cached in the container's tmp storage for the lifetime of the instance.
EFFNET_MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    "discogs-effnet-bs64-1.pb"
)

SAMPLE_RATE = 16000

# Cap how much audio we actually embed. The offline pipeline mean-pools over
# the whole track, but on a single shared CPU core a 7-minute track takes
# far longer than anyone will wait. A 2-minute window taken from the middle
# of the track skips intros/outros and lands on the part that actually
# characterises the genre — predictions match the full-track ones closely.
ANALYSIS_SECONDS = 120
MIN_SECONDS = 5

PORTFOLIO_URL = "https://thijnbakker.github.io/projects/dj-genre-classification.html"

st.set_page_config(
    page_title="DJ Genre Classifier",
    page_icon="🎧",
    layout="centered",
)

# These must be set before essentia (and therefore TensorFlow) is imported.
# TensorFlow allocates a scratch arena per worker thread, so on the single
# shared core a Community Cloud container gets, extra threads buy no speed
# and cost a lot of memory: measured peak RSS drops from ~1.34GB to ~0.96GB
# when pinned to one thread, while a 120s track still embeds in ~2s.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF's stderr banner
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no GPU on Cloud anyway
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_var, "1")


# ─── Cached resource loading ────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_classifier():
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(LABEL_ENC_PATH, "rb") as f:
        label_enc = pickle.load(f)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return clf, label_enc, metadata


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Downloads (once, cached on disk) and loads the Essentia
    Discogs-EffNet feature extractor used to produce 1280-dim embeddings."""
    import essentia.standard as es

    cache_path = Path(tempfile.gettempdir()) / "discogs-effnet-bs64-1.pb"
    if not cache_path.exists():
        # Download to a sibling temp file and rename, so an interrupted
        # download can never leave a truncated .pb in the cache.
        partial = cache_path.with_suffix(".pb.part")
        urllib.request.urlretrieve(EFFNET_MODEL_URL, partial)
        partial.replace(cache_path)

    return es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(cache_path),
        output="PartitionedCall:1",  # embeddings layer, not the 400-class head
    )


def extract_embedding(audio_path, emb_model):
    """Mirrors the real project's extract_embeddings.py, but on a bounded
    window of audio. Returns (embedding, error_message, seconds_analysed)."""
    import essentia.standard as es

    try:
        audio = es.MonoLoader(
            filename=str(audio_path), sampleRate=SAMPLE_RATE, resampleQuality=4
        )()
    except Exception:
        return None, "Couldn't decode that file — it may be corrupt or an unsupported codec.", 0

    if len(audio) < SAMPLE_RATE * MIN_SECONDS:
        return None, f"Track is shorter than {MIN_SECONDS} seconds — need more audio to get a reliable embedding.", 0

    # Centre window, so we skip the intro and outro of a full-length track.
    window = SAMPLE_RATE * ANALYSIS_SECONDS
    if len(audio) > window:
        start = (len(audio) - window) // 2
        audio = audio[start:start + window]

    try:
        embeddings = emb_model(audio)
    except Exception:
        return None, "The embedding model failed on this file. Try a different track.", 0

    if embeddings.size == 0:
        return None, "Essentia couldn't extract any frames from this file.", 0

    return embeddings.mean(axis=0).astype(np.float32), None, len(audio) / SAMPLE_RATE


# ─── UI ──────────────────────────────────────────────────────────────

st.title("🎧 DJ Genre Classifier")
st.markdown(
    "Upload a track and this runs the **actual model** trained for my "
    f"[DJ Genre Classification project]({PORTFOLIO_URL}) — the same "
    "classifier that powers the review app in that write-up, just without "
    "your music library attached to it."
)

with st.expander("How this works / what you're looking at", expanded=False):
    st.markdown(
        f"""
1. Your file is loaded and resampled to {SAMPLE_RATE // 1000}kHz mono, and a
   {ANALYSIS_SECONDS}-second window from the middle of the track is taken.
2. **Essentia's Discogs-EffNet** model turns it into a 1280-dimensional embedding —
   this is the same pretrained feature extractor the real pipeline uses.
3. A **logistic regression classifier** (trained on ~2,600 tracks I hand-labeled
   from my own library, with sigmoid-calibrated confidence) predicts a genre
   from that embedding.
4. Nothing is saved. The file lives in memory for this request only.

The model knows 38 genres, spanning House, Techno, Trance, Breaks & Bass,
Classics & Roots, and Mainstream & Pop. It's a genuinely honest model, not
a polished one — some genres it nails, some it's mediocre at, and it'll
tell you which is which below.
        """
    )

uploaded = st.file_uploader(
    f"Upload a track (MP3, WAV, or FLAC — at least {MIN_SECONDS} seconds long)",
    type=["mp3", "wav", "flac"],
)

if uploaded is not None:
    clf, label_enc, metadata = load_classifier()

    with st.spinner("Loading the audio embedding model (first run only takes longer)..."):
        emb_model = load_embedding_model()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded.name).suffix, delete=False
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.spinner("Extracting audio embedding..."):
            embedding, error, analysed = extract_embedding(tmp_path, emb_model)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if error:
        st.warning(error)
    else:
        proba = clf.predict_proba(embedding.reshape(1, -1))[0]
        order = np.argsort(-proba)
        top1_idx = order[0]
        top1_genre = label_enc.classes_[top1_idx]
        top1_conf = proba[top1_idx]

        st.subheader(f"Prediction: {top1_genre}")
        st.metric("Confidence", f"{top1_conf*100:.1f}%")

        st.markdown("**Top 3 candidates:**")
        for idx in order[:3]:
            genre = label_enc.classes_[idx]
            conf = proba[idx]
            st.progress(min(float(conf), 1.0), text=f"{genre} — {conf*100:.1f}%")

        # Per-class reliability context, pulled from the real training report
        report = metadata.get("classification_report", {})
        class_stats = report.get(top1_genre)
        if class_stats:
            f1 = class_stats["f1-score"]
            support = int(class_stats["support"])
            if f1 >= 0.75:
                note = "this is one of the model's **strongest** genres"
            elif f1 >= 0.5:
                note = "this is a **moderately reliable** genre for the model"
            else:
                note = "this genre is genuinely **hard for the model** — take the prediction with a grain of salt"
            st.caption(
                f"On held-out cross-validation, **{top1_genre}** scored an F1 of "
                f"{f1:.2f} (trained on {support} labeled examples) — {note}."
            )

        st.caption(f"Analysed {analysed:.0f} seconds of audio from the middle of the track.")

st.divider()
st.caption(
    "Built by Thijn Bakker · "
    f"[Full project write-up]({PORTFOLIO_URL}) · "
    "[Portfolio](https://thijnbakker.github.io)"
)
