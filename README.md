# DJ Genre Classifier — Live Demo

A live, interactive demo of the genre classifier built for my [DJ Genre
Classification project](https://thijnbakker.github.io/projects/dj-genre-classification.html).

Upload any track and it runs the **actual trained model** — a logistic
regression classifier with sigmoid-calibrated confidence, trained on
~2,600 hand-verified labels across 38 genres, sitting on top of Essentia's
Discogs-EffNet audio embeddings. It's the same classifier that powers the
"likely misclassified" flags in the review app described in the write-up —
just without a 7,600-track private library attached to it.

**Try it live:** https://dj-genre-classifier.streamlit.app

## How it works

1. Your uploaded file is loaded and resampled to 16kHz mono, and a
   **120-second window from the middle of the track** is taken.
2. Essentia's pretrained **Discogs-EffNet** model (18MB, downloaded once and
   cached for the container's lifetime) turns it into a 1280-dimensional
   embedding.
3. A **scikit-learn logistic regression classifier**, wrapped in
   `CalibratedClassifierCV` for honest confidence scores, predicts a genre.
4. Nothing is saved — the file exists only in memory for the request.

The offline pipeline (`extract_embeddings.py` in the main project repo)
mean-pools over the *whole* track. The demo windows to 120s because a shared
single-core container shouldn't spend 30 seconds on a 9-minute track. Each
result says how much audio it actually analysed.

## Layout

```
app.py                     the demo
model/classifier.pkl       trained CalibratedClassifierCV (copied from
model/label_encoder.pkl    dj_classifier_v2/ in the main project repo —
model/metadata.json        do not retrain here)
.streamlit/config.toml     portfolio-matching theme + 50MB upload cap
requirements.txt           pinned; see the warning below
packages.txt               ffmpeg, libsndfile1 (apt packages for Cloud)
```

## Running locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the 18MB Essentia embedding model into the system temp
directory — subsequent runs reuse it.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing
   at `app.py`.
3. **In Advanced settings, set the Python version to 3.12** (or 3.9–3.13).
   This matters — see below.
4. `packages.txt` and `requirements.txt` are picked up automatically.

### Why the pins are load-bearing

- **`essentia-tensorflow==2.1b6.dev1389`** — the *latest* release
  (`dev1438`) publishes **cp314 wheels only**. Leaving this unpinned means pip
  finds no installable wheel on any Python version Streamlit Cloud offers, and
  the build fails. `dev1389` is the newest release with cp39–cp313 manylinux
  wheels.
- **`scikit-learn==1.8.0`** — the version `classifier.pkl` was pickled with.
  A different minor version unpickles with warnings and can silently change
  behaviour. If you retrain, re-pin this to match.
- **`numpy<2`** — the TensorFlow runtime bundled inside `essentia-tensorflow`
  is built against the numpy 1.x ABI.

### Resource headroom

Measured locally on Python 3.12, after serving one prediction:

| | |
|---|---|
| Peak RSS | **1.03 GB** |
| Steady RSS | 877 MB |
| Inference, 120s audio, 1 core | ~2.1s |
| `essentia-tensorflow` wheel | 291 MB |

`app.py` pins TensorFlow to a single thread before importing Essentia.
TF allocates a scratch arena per worker thread, so on a one-core container the
extra threads buy no speed and cost real memory — peak RSS drops from
**~1.34 GB to ~0.96 GB** with the pin, which is the difference between fitting
and not. Don't remove those `os.environ` lines.

At ~1 GB peak this fits current Community Cloud containers, but not by a wide
margin. If the app starts getting OOM-killed, in rough order of preference:
lower `ANALYSIS_SECONDS` in `app.py`, move the embedding step to a small
separate API and keep this as a thin client, or host on Hugging Face Spaces
(16GB on the free CPU tier) instead.

## What this is (and isn't)

This is a research/portfolio demo, not a production service. The classifier
is genuinely uneven across genres — strong on sonically distinct classes
like Hard Groove and Hard Techno, weaker on blurry-boundary genres like
Deep House vs. Progressive House vs. Melodic House. The app surfaces that
honestly: each prediction comes with the model's actual cross-validated
F1 score for that genre, not just a bare confidence number.

## Related

- [Full project write-up](https://thijnbakker.github.io/projects/dj-genre-classification.html)
- [Portfolio](https://thijnbakker.github.io)
