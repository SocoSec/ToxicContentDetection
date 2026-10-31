# Toxic Content Detection Platform

A multi-label toxic content detection system that identifies **hate speech** and **cyberbullying** in text using a fine-tuned RoBERTa transformer model. Built with adversarial robustness testing and a REST API demo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SocoSec/ToxicContentDetection/blob/main/notebooks/train_on_colab.ipynb)

> **Want the full story?** See [`WALKTHROUGH.txt`](WALKTHROUGH.txt) for a step-by-step explanation of the architecture, the transformer / transfer-learning approach, and the results. To train on a free GPU for real metrics, click the Colab badge above.

---

## Project Structure

```
ToxicContentDetection/
├── data/
│   ├── raw/              # Original downloaded datasets
│   └── processed/        # Cleaned, tokenized, split datasets
├── models/               # Saved model checkpoints
├── src/
│   ├── dataset_loader.py  # Dataset loading and merging
│   ├── preprocess.py      # Text cleaning pipeline
│   ├── train.py           # Model training (fine-tuned RoBERTa)
│   └── predict.py         # Inference helper
├── evaluation/
│   └── evaluate.py        # F1, precision, recall, confusion matrix
├── adversarial/
│   └── adversarial_test.py # Robustness tests (leetspeak, emoji, typos)
├── demo/
│   └── app.py             # Flask REST API demo
├── notebooks/             # Jupyter exploration notebooks
├── requirements.txt
└── README.md
```

---

## Labels

| Label | Description |
|---|---|
| `hate_speech` | Content targeting groups based on race, religion, gender, etc. |
| `cyberbullying` | Personal attacks, threats, or harassment targeting individuals |
| `offensive` | Offensive language not directed at a specific target |
| `clean` | Normal, non-toxic content |

---

## Datasets Used

- **Davidson et al. (2017)** — Hate speech and offensive language (Twitter)
- **Kaggle Cyberbullying Classification** — Cyberbullying types dataset
- **HatEval (SemEval 2019)** — Multilingual hate speech detection

---

## Model

- Base: `roberta-base` (HuggingFace Transformers)
- Task: Multi-label text classification
- Fine-tuned on combined dataset with class balancing

---

## Adversarial Tests

The model is tested against obfuscation attacks including:
- Leetspeak substitution (h4te, k1ll)
- Emoji injection (🖕, 💀)
- Character repetition (haaaate)
- Deliberate misspellings
- Unicode homoglyphs

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Load and preprocess data
python src/dataset_loader.py
python src/preprocess.py

# Train the model
python src/train.py

# Evaluate
python evaluation/evaluate.py

# Run adversarial tests
python adversarial/adversarial_test.py

# Start the API demo
python demo/app.py
```

---

## API Usage

```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "text": "I hate all people from that country"
}
```

Response:
```json
{
  "text": "I hate all people from that country",
  "predictions": {
    "hate_speech": 0.91,
    "cyberbullying": 0.12,
    "offensive": 0.76,
    "clean": 0.03
  },
  "label": "hate_speech"
}
```

---

## Author

Esmail Abdullah — Cybersecurity Engineering MSc, Óbuda University
