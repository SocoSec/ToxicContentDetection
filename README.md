# Toxic Content Detection Platform

A **multi-class** text classifier that flags **hate speech**, **cyberbullying**, and
**offensive language** using a fine-tuned RoBERTa transformer — shipped with a
cleaning + training pipeline, test-set evaluation, an **adversarial robustness
suite**, and a REST API.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SocoSec/ToxicContentDetection/blob/main/notebooks/train_on_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> New here? Read [`WALKTHROUGH.txt`](WALKTHROUGH.txt) for a plain-English, step-by-step
> explanation of the architecture and the transformer / transfer-learning approach.

---

## What it does

Given a sentence, the model predicts one of four labels with a confidence score:

| Label | Meaning |
|---|---|
| `clean` | Normal, non-toxic content |
| `offensive` | Offensive language not aimed at a specific target |
| `hate_speech` | Attacks a group by race, religion, gender, etc. |
| `cyberbullying` | Personal attacks, threats, or harassment of an individual |

It also **red-teams itself**: an adversarial suite measures how easily obfuscated
text (leetspeak, homoglyphs, emoji, spacing…) can evade the classifier.

---

## How it works

- **Model:** `roberta-base` (125M-parameter transformer, pre-trained on ~160GB of
  English text).
- **Approach:** *transfer learning* — the pre-trained model is **fine-tuned** on
  labelled toxic/clean examples with a small learning rate, plus a fresh 4-way
  classification head. Class-weighted cross-entropy handles label imbalance.
- **Why a transformer:** self-attention lets the model read each word in context
  (e.g. "do **not** hate" vs "hate"), which bag-of-words models miss.

Pipeline:

```
raw CSVs → dataset_loader.py → preprocess.py → train.py → models/toxic_roberta/
                                                   ↓
                              evaluate.py · adversarial_test.py · demo/app.py
```

---

## Quick start

Requires **Python 3.10+**. A GPU is optional (CPU works, just slower).

```bash
# 1. Clone
git clone https://github.com/SocoSec/ToxicContentDetection.git
cd ToxicContentDetection

# 2. Create an isolated environment and install deps
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. Get data (Davidson is public, no login — see data/README.md for the others)
#    Or skip: the loader falls back to a small synthetic sample so the pipeline
#    still runs with nothing downloaded.

# 4. Run the pipeline
python src/dataset_loader.py        # build the combined dataset
python src/preprocess.py            # clean + train/val/test split
python src/train.py                 # fine-tune RoBERTa -> models/toxic_roberta/
python evaluation/evaluate.py       # metrics + confusion matrix on the test set
python adversarial/adversarial_test.py   # robustness report

# 5. Serve the API
python demo/app.py                  # http://localhost:5000
```

**Fast CPU demo run** (no GPU): the training script accepts env-var overrides so
you can finish in minutes instead of hours —

```bash
# Windows PowerShell
$env:TOXIC_MAX_TRAIN=2000; $env:TOXIC_EPOCHS=1; $env:TOXIC_MAX_LENGTH=64; python src/train.py
```

**Full training on a free GPU:** click the **Open in Colab** badge above and
*Run all* — it clones, installs, downloads data, trains, and shows the results.

---

## API usage

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "What an absolute idiot, cant believe you said that"}'
```

Response (probabilities are a softmax over the four classes, so they sum to 1):

```json
{
  "text": "What an absolute idiot, cant believe you said that",
  "cleaned": "What an absolute idiot, cant believe you said that",
  "predictions": {
    "clean": 0.159,
    "offensive": 0.434,
    "hate_speech": 0.355,
    "cyberbullying": 0.052
  },
  "label": "offensive",
  "confidence": 0.434
}
```

Endpoints: `GET /health`, `POST /predict` (`{"text": "..."}`),
`POST /predict/batch` (`{"texts": ["...", "..."]}`).

---

## Results

> These are a **CPU smoke-test baseline** (1 epoch on 2,000 rows of real Davidson
> data + a synthetic cyberbullying top-up) that proves the pipeline end to end.
> Run the Colab notebook for full-data, multi-epoch GPU results — expect the
> macro-F1 to rise substantially.

**Classification (unseen test set, 3,738 rows):**

| Metric | Value |
|---|---|
| Accuracy | 0.879 |
| Macro-F1 | 0.45 |
| F1 `offensive` | 0.93 |
| F1 `clean` | 0.79 |
| F1 `hate_speech` | 0.09 |
| F1 `cyberbullying` | 0.00 |

Accuracy is flattered by the majority class (`offensive` ≈ 77% of the data);
macro-F1 is the honest number. The rare classes are under-learned in this short
run — more data + training + balancing fixes it.

**Adversarial robustness (overall score 66.5 / 100 — higher is better):**

| Attack | Evasion rate |
|---|---|
| Leetspeak (`h4t3`) | **78.4%** |
| Mixed case (`HaTe`) | 69.4% |
| Homoglyphs (Cyrillic `а`) | 52.8% |
| Whitespace (`h a t e`) | 13.2% |
| Misspelling (`haet`) | 11.2% |
| Char repetition (`haaate`) | 6.8% |
| Emoji injection | 2.6% |

**Takeaway:** the model shrugs off additive noise but is fragile to character
substitution — a real evader writes `k1ll`, not `kill`. The fix is adversarial
data augmentation and input normalisation (de-leet, map homoglyphs to ASCII,
lowercase) before inference.

---

## Project structure

```
ToxicContentDetection/
├── src/
│   ├── dataset_loader.py     # load + merge datasets into one label scheme
│   ├── preprocess.py         # text cleaning + stratified train/val/test split
│   ├── train.py              # fine-tune roberta-base (env-var knobs for quick runs)
│   └── predict.py            # CLI inference helper
├── evaluation/evaluate.py    # F1/precision/recall + confusion matrix
├── adversarial/adversarial_test.py  # obfuscation attacks + robustness score
├── demo/app.py               # Flask REST API
├── notebooks/train_on_colab.ipynb   # full GPU training on Colab
├── data/README.md            # how to download the datasets
├── requirements.txt
├── WALKTHROUGH.txt           # full step-by-step explanation
└── README.md
```

---

## Datasets

See [`data/README.md`](data/README.md) for download links.

- **Davidson et al. (2017)** — hate speech & offensive language (public, no login)
- **Kaggle Cyberbullying Classification** — cyberbullying types (free login)
- **HatEval (SemEval 2019)** — hate speech detection

---

## Limitations

- A content classifier is an **assistant to human moderation**, not a final
  verdict; it inherits the biases of its training data.
- Results above are a deliberately small CPU baseline — not a tuned model.
- The `cyberbullying` class currently uses synthetic placeholder text until the
  Kaggle set is added.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Esmail Abdullah** — Cybersecurity Engineering MSc, Óbuda University
