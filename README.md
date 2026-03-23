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

