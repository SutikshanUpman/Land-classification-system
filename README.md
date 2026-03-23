# 🌍 Agricultural Land Classification System

> A deep learning pipeline for satellite imagery binary classification using CNNs and Vision Transformers — built with Keras and PyTorch.

---

## 📌 Project Overview

This capstone project develops a **binary land classification system** for agricultural applications. Using satellite imagery, the system classifies land into two categories:

- 🌾 **class_1_agri** — Cultivable / agricultural land
- 🪨 **class_0_non_agri** — Non-cultivable / non-agricultural land

**Dataset:** 6,000 images total (3,000 per class), Sentinel-2 satellite tiles.  
**Split:** 70% train / 15% val / 15% test → 4,200 / 900 / 900 images.

The project runs **parallel implementations in Keras and PyTorch**, enabling direct framework comparison across all stages.

---

## 🗂️ Project Structure

```
land-classification/
│
├── data/
│   ├── raw/                    # Original satellite images — never modified
│   ├── processed/              # Resized to 64×64 RGB — NO normalization
│   └── splits/                 # train.csv / val.csv / test.csv — shared across all stages
│
├── stage1_data/
│   ├── memory_loader.py        # Bulk RAM loading (comparison only)
│   ├── generator_loader.py     # Path-based sequential loader (production)
│   ├── augmentation_keras.py   # Keras tf.data augmentation pipeline
│   └── augmentation_torch.py   # PyTorch transforms augmentation pipeline
│
├── stage2_cnn/
│   ├── cnn_keras.py
│   ├── cnn_pytorch.py
│   ├── evaluate.py
│   └── saved_models/
│
├── stage3_transformer/
│   ├── vit_keras.py
│   ├── vit_pytorch.py
│   ├── transfer_learning.py
│   └── saved_models/
│
├── stage4_final/
│   ├── comparison.py
│   └── report/
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🗃️ Data Directory — Roles

| Folder | Written by | Read by | Contents |
|--------|-----------|---------|----------|
| `data/raw/` | Never — source only | Stage 1 | Original satellite images, untouched |
| `data/processed/` | Stage 1 | Stage 2, 3 | Images resized to 64×64 RGB — no normalization |
| `data/splits/` | Stage 1 | Stage 2, 3, 4 | `train.csv`, `val.csv`, `test.csv` — paths + labels into `data/processed/` |

**Why separate `raw/` from `processed/`?**  
Raw images are inconsistent in size and format. Stage 1 handles resize and format conversion once. Normalization and augmentation are intentionally excluded — normalization is model-specific and augmentation must remain random per epoch, so both happen live during training.

```
Stage 1 — Preprocessing (done once, deterministic):
  ✔ Resize all images to 64×64
  ✔ Convert to RGB
  ✔ Skip corrupt images
  ✖ Do NOT normalize pixel values

Stage 2 & Stage 3 — Training-time (live, per epoch):
  ✔ Normalize pixel values to [0, 1]
  ✔ Random augmentation:
      - Horizontal / vertical flip
      - Random rotation
      - Color jitter
      - Random crop
```

---

### 🔒 Data Consistency Rule

All stages MUST load from the same CSV splits:

```
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

❌ Do NOT use framework-based splitting (e.g., `validation_split` in Keras) for training.  
`image_dataset_from_directory` with `validation_split` is used in Stage 1 **for demonstration only** to compare Keras built-in loading vs. the custom CSV-based approach. All training in Stages 2 & 3 must use the CSV splits.

**Reason:** Ensures identical data exposure across Keras and PyTorch, fair CNN vs. ViT comparison, and prevents data leakage.

---

## 🔄 Data Flow Across the Pipeline

```
data/raw/images_dataSAT/
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1 — Data Handling                    │
│                                             │
│  1. Explore raw data                        │
│     - Class balance, image dimensions       │
│     - Memory vs generator loading comparison│
│                                             │
│  2. Preprocess and save to disk (once)      │
│     - Resize all images to 64×64            │
│     - Save to data/processed/               │
│                                             │
│  3. Create reproducible splits              │
│     - Divide into train / val / test        │
│     - Save CSVs (paths → data/splits/)      │
│                                             │
│  4. Build and validate augmentation pipes   │
│     - Keras: tf.data + augmentation layers  │
│     - PyTorch: ImageFolder + transforms     │
└──────────┬──────────────────────────────────┘
           │
           ├──► data/processed/class_0_non_agri/    ┐
           ├──► data/processed/class_1_agri/        ├─ used for all training
           ├──► data/splits/train.csv               │
           ├──► data/splits/val.csv                 │
           └──► data/splits/test.csv               ─┘
                               │
           ┌───────────────────┴──────────────────────────────┐
           │                                                  │
           ▼                                                  ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Stage 2 — CNN               │         │  Stage 3 — Transformer       │
│                              │         │                              │
│  Loads split CSVs            │         │  Loads same split CSVs       │
│  Reads data/processed/       │         │  Reads data/processed/       │
│  Augments on-the-fly         │         │  Augments on-the-fly         │
│  Trains CNN (Keras+PyTorch)  │         │  Fine-tunes ViT (Keras+      │
│  Saves weights →             │         │  PyTorch)                    │
│  stage2_cnn/saved_models/    │         │  Saves weights →             │
└──────────┬───────────────────┘         │  stage3_transformer/         │
           │                             │  saved_models/               │
           │                             └──────────┬───────────────────┘
           └──────────────────┬───────────────────  ┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 4 — Final Report     │
                │                             │
                │  Loads all saved weights    │
                │  Evaluates on test split    │
                │  Compares CNN vs ViT        │
                │  Compares Keras vs PyTorch  │
                │  Produces final report      │
                └─────────────────────────────┘
```

---

## 🧩 Pipeline Stages

### Stage 1 — Data Handling

| Step | What happens | Output |
|------|-------------|--------|
| Exploration | Visualize samples, check class balance | Understanding of raw data |
| Loading comparison | RAM-based (37.94 MB, 2.95s) vs path-based (0.54 MB, 0.03s) | Justification for generator approach |
| Preprocessing | Resize to 64×64 RGB, skip corrupt images | `data/processed/` |
| Split creation | 70/15/15 stratified split, saved as CSVs | `data/splits/*.csv` |
| Keras pipeline | Custom generator + `image_dataset_from_directory` (demo) + augmentation + `.cache().prefetch()` | Keras data pipeline |
| PyTorch pipeline | `ImageFolder` + `transforms.Compose` + `DataLoader` | PyTorch data pipeline |

**Key files:** `memory_loader.py`, `generator_loader.py`, `augmentation_keras.py`, `augmentation_torch.py`

---

### Stage 2 — CNN Development

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Load split CSVs, read from `data/processed/` | `data/splits/`, `data/processed/` | Batched train/val/test |
| Augmentation | Random flips, rotations — live each epoch | Preprocessed batches | Augmented training batches |
| Model training | Custom CNN in Keras and PyTorch | Augmented batches | Trained weights |
| Evaluation | Accuracy, Precision, Recall, F1, AU-ROC on test set | Test split + weights | Metrics for Stage 4 |

**Key files:** `cnn_keras.py`, `cnn_pytorch.py`, `evaluate.py`

---

### Stage 3 — Vision Transformer Integration

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Same CSV splits and `data/processed/` as Stage 2 | `data/splits/`, `data/processed/` | Batched train/val/test |
| Fine-tuning | Pre-trained ViT adapted in Keras and PyTorch | Augmented batches | Trained ViT weights |
| Evaluation | Same metrics on same test split as Stage 2 | Test split + weights | Metrics for Stage 4 |

**Key files:** `vit_keras.py`, `vit_pytorch.py`, `transfer_learning.py`

---

### Stage 4 — Final Report

| Step | What happens | Input |
|------|-------------|-------|
| Model loading | Load all saved weights from Stages 2 & 3 | `saved_models/` |
| Final evaluation | All models on the held-out test split | `data/splits/test.csv` + `data/processed/` |
| Comparison | CNN vs ViT, Keras vs PyTorch — metrics table + plots | Evaluation results |
| Report | Findings, trade-offs, recommendations | All above |

**Key files:** `comparison.py`, `report/`

---

## 📊 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall classification correctness |
| Precision | How often positive predictions are correct |
| Recall | How many actual positives were captured |
| F1-Score | Harmonic mean of Precision and Recall |
| AU-ROC | Model's ability to distinguish between classes |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | Keras / TensorFlow 2.20.0, PyTorch |
| Data Processing | NumPy, Pandas, Pillow |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter / VS Code |
| Evaluation | Scikit-learn |

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/land-classification.git
cd land-classification
pip install -r requirements.txt
```

---

## 📈 Results Summary *(to be updated)*

| Model | Framework | Accuracy | F1-Score | AU-ROC |
|-------|-----------|----------|----------|--------|
| CNN | Keras | — | — | — |
| CNN | PyTorch | — | — | — |
| ViT | Keras | — | — | — |
| ViT | PyTorch | — | — | — |

---

## 👤 Author

**AI Engineer** — Fertilizer Company Land Intelligence Team  
*Capstone Project — Deep Learning for Agricultural Applications*

## 📄 License

This project is for academic and internal use only.
