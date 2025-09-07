# Quran Tafsir Search - Streamlit App

A semantic search application for Quran commentary (tafsir) using ChromaDB and Streamlit.

## Features

- Semantic search through Quran commentary
- Filter by specific Surah (chapter)
- Arabic text display with audio recitation
- Dark mode optimized interface

## Deployment

This is a streamlined version containing only the files needed for deployment:

- `streamlit_app.py` - Main application (self-contained)
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `.streamlit/` - Streamlit configuration
- `chroma_db/` - Pre-computed embeddings database
- `quran-data/` - Quran text and metadata

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Set main file path to `streamlit_app.py`
5. Deploy!
