# YelpZIP Fake Review Detector

End-to-end NLP project that detects fake Yelp reviews using six models across
three paradigms, served through a clean web interface.

---

## Project Structure

```
yelpzip_project/
├── data_prep.py            # Step 1 – preprocessing, splits, shared artefacts
├── classical_models.py     # Step 2A – Logistic Regression & Random Forest
├── deep_models.py          # Step 2B – LSTM & BiLSTM
├── transformer_models.py   # Step 2C – BERT & RoBERTa fine-tuning
├── app.py                  # Flask web app
├── templates/
│   └── index.html          # Frontend UI
├── requirements.txt
└── artifacts/              # Created automatically – all saved models live here
```

---

## Quickstart

### 1 · Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Place your dataset

Put `yelpzip.csv` in the project root.  
Expected columns: `text`, `label` (values: `fake` / `real`  or  `-1` / `1`).

### 3 · Run the pipeline in order

```bash
# Prepare data – MUST run first
python data_prep.py

# Train classical models (fast, CPU-only)
python classical_models.py

# Train deep learning models (GPU recommended)
python deep_models.py

# Fine-tune transformers (GPU strongly recommended)
python transformer_models.py
```

### 4 · Run the web app locally

```bash
python app.py
# open http://localhost:5000
```

---

## Consistency Design

All model files share the same pre-processing decisions made once in `data_prep.py`:

| Artefact | Used by |
|---|---|
| `tfidf_vectorizer.pkl` | Logistic Regression, Random Forest |
| `tokenizer.pkl` + `X_*_seq.npy` | LSTM, BiLSTM |
| `train.csv` / `test.csv` (clean_text col) | BERT, RoBERTa |
| `y_train.npy` / `y_test.npy` | All models |
| `MAX_SEQ_LEN = 200` | Deep & Transformer models |
| `MAX_FEATURES = 30 000` | TF-IDF & Keras tokenizer |

---

## Free Deployment Options

### Option A – Render (recommended, zero config)

1. Push the project to a **GitHub repository**.
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo.
3. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
4. Add environment variable `PORT=10000` (Render injects this automatically).
5. Upload / commit your `artifacts/` folder (or use Render's persistent disk).

> Free tier sleeps after 15 minutes of inactivity – first request wakes it up.

---

### Option B – Hugging Face Spaces (great for ML projects)

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **SDK: Gradio** or **Docker**.  
   For Flask, choose **Docker** and add the `Dockerfile` below.
3. Push your code + `artifacts/` to the Space's Git repo.

```dockerfile
# Dockerfile (place in project root)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
```

---

### Option C – Railway

1. Install Railway CLI: `npm install -g @railway/cli`
2. `railway login` → `railway init` → `railway up`
3. Railway auto-detects Flask and runs `gunicorn app:app`.

---

## Notes

- **Classical models** run instantly on CPU – great for live demos.
- **LSTM / BiLSTM** load in ~2 s on CPU; first inference is ~200 ms.
- **BERT / RoBERTa** are large (~500 MB each); free tiers with 512 MB RAM  
  may struggle – consider using only classical or deep models for free hosting.
- You can selectively disable transformer models by simply not training them;  
  the app will return a friendly error if the artefact folder is missing.
