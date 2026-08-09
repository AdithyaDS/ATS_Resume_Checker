---
title: ATS Resume Matcher
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# ATS Resume Matcher

An ATS-style tool that scores a resume against a job description using semantic
similarity (sentence-transformers) + skill keyword matching (skillNer), and
generates AI-powered improvement suggestions via an LLM (Groq).

## Run locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md   # only if the wheel in requirements.txt fails
streamlit run app.py
```

You'll need a free Groq API key from https://console.groq.com — paste it into
the sidebar when the app opens (it is not stored anywhere).

## Deploy for free — Streamlit Community Cloud

Hugging Face Spaces now requires a paid PRO plan to run any Python-backed app
(Gradio/Docker) — only static, no-backend Spaces are free there. So the free
path for this project is **Streamlit Community Cloud**:

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app."
3. Point it at your repo, branch, and `app.py`, then Deploy.
4. First load takes a few minutes while it installs dependencies and downloads models.

**Free tier limits to know:** ~1 GB RAM, app sleeps after ~12 hours of no traffic
(just reopens on next visit), only 1 private app allowed (unlimited public apps).

## Why en_core_web_md instead of en_core_web_lg

`en_core_web_lg` (587 MB) plus `sentence-transformers` risks exceeding the 1 GB
free RAM limit. `en_core_web_md` (40 MB) still includes word vectors — so
skillNer's fuzzy/confidence-based skill matching still works — just with less
precise vectors than the large model. If you deploy somewhere with more RAM
later (a paid tier, your own server, etc.), you can switch back to
`en_core_web_lg` for slightly better accuracy.

## Notes

- Model loading is cached with `@st.cache_resource`, so it's slow only on the
  very first request after a restart, not on every analysis.
- The Groq API key is entered per-session in the sidebar rather than hardcoded,
  so you can share the deployed app without exposing your own key.
