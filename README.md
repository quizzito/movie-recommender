# 🎬 Movie Recommendation System
### Collaborative Filtering on MovieLens 25M

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=flat-square&logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-green?style=flat-square&logo=scikit-learn)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Training-F9AB00?style=flat-square&logo=googlecolab)

---

## 📌 Overview

An end-to-end movie recommendation engine built using multiple collaborative filtering approaches on the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset — **25 million ratings** from **162,000 users** across **62,000 movies**.

Models range from a memory-based baseline to a neural network, combined into an optimized ensemble that achieves an **RMSE of 0.7859** — outperforming every individual model.

---

## 🏆 Model Leaderboard

| Rank | Model | Type | RMSE | MAE |
|:----:|-------|------|:----:|:---:|
| 🥇 | **Weighted Ensemble** | Blending | **0.7859** | **0.5901** |
| 🥈 | Stacking Ensemble | Ridge Meta-Learner | 0.7863 | 0.5917 |
| 🥉 | SVD | Matrix Factorization | 0.7927 | 0.5949 |
| 4 | Simple Average | Blending | 0.7951 | 0.5992 |
| 5 | NCF | Neural CF (Colab GPU) | 0.8245 | 0.6202 |
| 6 | NMF | Matrix Factorization | 0.8873 | 0.6689 |
| 7 | User-Based CF | Memory-Based Baseline | 0.9629 | 0.7443 |
| 8 | ALS | Implicit Feedback | 1.3956 | 1.0357 |

> **Metric:** RMSE on a temporally split test set (last 20% of each user's ratings). Lower is better.

---

## 📊 Key Findings

- **Weighted Ensemble** is the best model — combining SVD, NMF, and NCF predictions with optimized weights outperforms every individual model
- **SVD** is the strongest single model — simple, fast, and hard to beat even with deep learning
- **NCF** trained on only 3 epochs (Google Colab T4 GPU) still outperforms NMF — more epochs would push it higher
- **ALS** underperforms on explicit rating prediction — it is designed for implicit feedback (clicks, views) and is better evaluated with Precision@K than RMSE
- Ensemble methods provide a **free improvement** of ~0.007 RMSE over the best single model with no additional data

---

## 🧠 Models Explained

### 1. User-Based Collaborative Filtering (Baseline)
Computes cosine similarity between users on the user-item rating matrix. Predicts ratings as a K-nearest-neighbour weighted average. Simple and interpretable but doesn't scale — similarity matrix is O(n²) in users.

### 2. SVD — Simon Funk Matrix Factorization
Decomposes the user-item matrix into latent user and item vectors using SGD. Includes bias terms for users, items, and global mean. The approach popularised by the Netflix Prize. Fast, robust, and consistently strong.

### 3. NMF — Non-Negative Matrix Factorization
Similar to SVD but constrains all latent factors to be non-negative. Makes learned components more interpretable — each factor loosely corresponds to a genre cluster or viewing style.

### 4. NCF — Neural Collaborative Filtering ⚡ Google Colab
Combines two neural pathways into one model:
- **GMF branch** — element-wise product of user & item embeddings (generalises matrix factorisation)
- **MLP branch** — concatenated embeddings through hidden layers [128 → 64 → 32] for non-linear interactions

Final prediction merges both branches through a sigmoid-scaled output layer. Trained on a **T4 GPU** via Google Colab due to compute requirements.

### 5. ALS — Alternating Least Squares ⚡ Google Colab
Treats ratings as implicit confidence signals rather than explicit preferences. Alternates between solving for user factors (item factors fixed) and item factors (user factors fixed). Best suited for implicit datasets (plays, clicks, purchases) where absence of a rating is meaningful. Trained on Google Colab CPU.

### 6. Weighted Ensemble 🏆
Learns optimal blend weights across SVD, NMF, and NCF using `scipy.optimize.minimize` with RMSE as the objective. Weights are constrained to be non-negative and sum to 1. Finds that SVD deserves the most trust, followed by NCF, then NMF.

### 7. Stacking Ensemble
Trains a Ridge regression meta-model where inputs are the three models' predictions and the target is the actual rating. Learns a linear combination with an intercept, adding slightly more flexibility than fixed weights.

---

## 🗂️ Project Structure

```
movie-recommender/
│
├── data/
│   ├── raw/                    # MovieLens 25M CSVs (gitignored)
│   └── processed/              # Temporal train/test split as parquet (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline.ipynb       # User-Based CF + data preprocessing
│   ├── 03_svd.ipynb            # SVD with hyperparameter tuning
│   ├── 04_nmf.ipynb            # NMF with latent factor visualisation
│   ├── 05_ncf.ipynb            # Neural CF (run locally or on Colab)
│   ├── 06_als.ipynb            # ALS implicit feedback model
│   ├── 07_evaluation.ipynb     # Cross-model comparison & charts
│   └── 08_ensembling.ipynb     # Weighted blend + stacking ensemble
│
├── models/                     # Saved model artifacts (gitignored)
├── reports/
│   └── figures/                # EDA plots, training curves, leaderboard charts
│
├── .env.example                # Credential template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11
- Mac (Intel or M1/M2), Linux, or Windows

### 1. Clone the repo
```bash
git clone https://github.com/quizzito/movie-recommender.git
cd movie-recommender
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-surprise --no-build-isolation   # build separately
```

### 4. Get the data
```bash
# Download directly from GroupLens (official source — no login required)
curl -L https://files.grouplens.org/datasets/movielens/ml-25m.zip -o data/raw/ml-25m.zip
unzip data/raw/ml-25m.zip -d data/raw/
mv data/raw/ml-25m/* data/raw/
rm -rf data/raw/ml-25m/ data/raw/ml-25m.zip
```

### 5. Launch Jupyter
```bash
python -m ipykernel install --user --name=movie-rec --display-name "Movie Recommender"
jupyter notebook
```

> In VS Code: open any `.ipynb` → top right → **Select Kernel → Movie Recommender**

---

## 🚀 Running the Notebooks

Run notebooks **in order**. Each one saves its outputs so the next can load them without reprocessing.

| Notebook | Where to run | Output saved |
|----------|:------------:|-------------|
| `01_eda.ipynb` | Local | Figures in `reports/figures/` |
| `02_baseline.ipynb` | Local | `train.parquet`, `test.parquet`, `baseline_cf.joblib` |
| `03_svd.ipynb` | Local | `svd.joblib` |
| `04_nmf.ipynb` | Local | `nmf.joblib` |
| `05_ncf.ipynb` | ⚡ Google Colab | `ncf.pt`, `ncf_meta.joblib` |
| `06_als.ipynb` | ⚡ Google Colab | `als.joblib` |
| `07_evaluation.ipynb` | Local | `leaderboard.csv` |
| `08_ensembling.ipynb` | Local | `final_leaderboard.csv`, `final_ensemble.joblib` |

---

## ⚡ Google Colab Setup (NCF & ALS)

NCF and ALS are computationally heavy and were trained on Google Colab's free GPU/CPU.

### Step 1 — Upload processed data to Colab
After running `02_baseline.ipynb` locally, upload the split files directly to Colab:

```python
from google.colab import files
import shutil, os

os.makedirs("/content/data/processed", exist_ok=True)
os.makedirs("/content/models", exist_ok=True)

# Upload train.parquet when prompted
files.upload()
shutil.move("/content/train.parquet", "/content/data/processed/train.parquet")

# Upload test.parquet when prompted
files.upload()
shutil.move("/content/test.parquet", "/content/data/processed/test.parquet")
```

### Step 2 — Enable GPU in Colab
```
Runtime → Change Runtime Type → Hardware Accelerator → T4 GPU → Save
```

### Step 3 — Install packages
```python
!pip install scikit-surprise implicit --no-build-isolation -q
```

### Step 4 — Set local paths
```python
import os
PROC_DIR  = "/content/data/processed/"
MODEL_DIR = "/content/models/"
```

### Step 5 — Train NCF and ALS, then download outputs
```python
# After training completes
from google.colab import files
files.download("/content/models/ncf_meta.joblib")
files.download("/content/models/ncf.pt")
files.download("/content/models/als.joblib")
```

### Step 6 — Move downloaded files to your local `models/` folder
```bash
mv ~/Downloads/ncf_meta.joblib models/
mv ~/Downloads/ncf.pt          models/
mv ~/Downloads/als.joblib      models/
```

Now run `07_evaluation.ipynb` and `08_ensembling.ipynb` locally — they load all model files automatically.

> **Colab tip:** Keep the browser tab active during training. Colab disconnects after ~90 minutes of inactivity. NCF (3 epochs) finishes in ~10 min on T4 GPU. ALS (10 iterations) takes ~15 min on CPU.

---

## 🛠️ Tech Stack

| Category | Library |
|----------|---------|
| Data processing | `pandas`, `numpy`, `scipy` |
| Machine learning | `scikit-learn`, `scikit-surprise` |
| Deep learning | `PyTorch` |
| Implicit feedback | `implicit` |
| Visualisation | `matplotlib`, `seaborn` |
| Serialisation | `joblib` |
| Environment | `python-dotenv` |
| Compute (heavy models) | Google Colab T4 GPU |

---

## 📈 Training Configuration

| Model | Key Parameters |
|-------|---------------|
| User-Based CF | K=50 neighbours, cosine similarity |
| SVD | 100 factors, 20 epochs, lr=0.005, reg=0.02 |
| NMF | 15 factors, 50 epochs, reg_pu=0.06 |
| NCF | embed_dim=64, hidden=[128,64,32], 3 epochs, batch=1024 |
| ALS | 50 factors, 10 iterations, α=40.0 |
| Weighted Ensemble | Weights optimised via `scipy.minimize` (SLSQP) |
| Stacking Ensemble | Ridge regression, α=1.0 |

---

## 🔑 Next Improvements

- [ ] Increase NCF to 10+ epochs on Colab for better accuracy
- [ ] Add temporal bias — users rate more generously in earlier years
- [ ] Hybrid model — inject genre embeddings into NCF (content + CF)
- [ ] Two-stage pipeline — ALS for fast candidate retrieval → NCF for re-ranking
- [ ] `predict_for_user()` deployment function for live recommendations
- [ ] LightFM — hybrid model combining collaborative + content features

---

## 👤 Author

GitHub: [@quizzito](https://github.com/quizzito)

---

