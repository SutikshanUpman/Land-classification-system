# 🌍 Agricultural Land Classification System

> A deep learning pipeline for satellite imagery classification using CNNs and Vision Transformers — built with Keras and PyTorch.

---

## 📌 Project Overview

This capstone project develops a **multi-class terrain classification system** for agricultural applications. Using satellite imagery, the system identifies and classifies different land cover types:

- 🌾 Croplands
- 🌲 Forests
- 💧 Water Bodies
- 🏙️ Urban Areas
- 🪨 Barren Land

The project runs **parallel implementations in Keras and PyTorch**, enabling direct framework comparison across all stages of the pipeline.

---

## 🗂️ Project Structure

```
land-classification/
│
├── data/
│   ├── raw/                    # Original satellite images — never modified
│   ├── processed/              # Resized + normalized images — written once by Stage 1,
│   │                           # read by Stage 2 and Stage 3 for training
│   └── splits/                 # train.csv / val.csv / test.csv — shared across all stages
│
├── stage1_data/
│   ├── memory_loader.py        # Demonstrates bulk loading into RAM (comparison only)
│   ├── generator_loader.py     # Path-based sequential loader (production approach)
│   ├── augmentation_keras.py   # Keras tf.data augmentation pipeline
│   ├── augmentation_torch.py   # PyTorch transforms augmentation pipeline
│   └── geo_dataloader.py       # Custom geospatial-aware data loader
│
├── stage2_cnn/
│   ├── cnn_keras.py            # CNN model architecture in Keras
│   ├── cnn_pytorch.py          # CNN model architecture in PyTorch
│   ├── evaluate.py             # Accuracy, Precision, Recall, F1-Score evaluation
│   └── saved_models/           # Trained CNN weights saved here
│
├── stage3_transformer/
│   ├── vit_keras.py            # Vision Transformer (Keras)
│   ├── vit_pytorch.py          # Vision Transformer (PyTorch)
│   ├── transfer_learning.py    # Fine-tuning pre-trained ViT models
│   └── saved_models/           # Trained ViT weights saved here
│
├── stage4_final/
│   ├── comparison.py           # CNN vs ViT comparative analysis
│   └── report/                 # Final project report and visualizations
│
├── notebooks/                  # One Jupyter notebook per stage
├── requirements.txt
└── README.md
```

---

## 🗃️ Data Directory — Roles Explained

| Folder | Written by | Read by | Contents |
|--------|-----------|---------|----------|
| `data/raw/` | Never — source data only | Stage 1 only | Original satellite images, untouched |
| `data/processed/` | Stage 1 | Stage 2, Stage 3 | Images resized to 64×64 and normalized — deterministic steps done once |
| `data/splits/` | Stage 1 | Stage 2, Stage 3, Stage 4 | `train.csv`, `val.csv`, `test.csv` — file paths + labels pointing to `data/processed/` |

### Why separate `raw/` from `processed/`?

Raw satellite images are often inconsistent in size and format. Every training epoch would otherwise repeat the same deterministic operations — resize, decode, normalize — at significant CPU cost. Stage 1 performs these steps **once**, saves the clean output to `data/processed/`, and all training stages read from there.

**Random augmentation** (flips, rotations, color jitter) is intentionally **not** pre-saved — it must remain random and different each epoch. It is applied live inside the DataLoader or `tf.data` pipeline during training.

```
Stage 1 pre-saves to data/processed/ (deterministic, done once):
  ✔ Resize all images to 64×64
  ✔ Normalize pixel values to [0, 1]
  ✔ Convert to consistent format
  ✔ Remove or flag corrupt images

Applied on-the-fly during Stage 2 and Stage 3 training (random, every epoch):
  ✔ Random horizontal / vertical flip
  ✔ Random rotation
  ✔ Color jitter
  ✔ Random crop
```

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
│     - Normalize pixel values to [0, 1]      │
│     - Save to data/processed/               │
│                                             │
│  3. Create reproducible splits              │
│     - Divide into train / val / test        │
│     - Save CSVs (paths → data/processed/)   │
│                                             │
│  4. Build and validate augmentation pipes   │
│     - Keras: tf.data + augmentation layers  │
│     - PyTorch: ImageFolder + transforms     │
└──────────┬──────────────────────────────────┘
           │
           ├──► data/processed/class_0_non_agri/   ┐
           ├──► data/processed/class_1_agri/        ├─ used for all training
           ├──► data/splits/train.csv               │
           ├──► data/splits/val.csv                 │
           └──► data/splits/test.csv               ─┘
                      │
           ┌──────────┴───────────────────────────────────────┐
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
**Goal:** Explore raw data, build preprocessing and augmentation pipelines, and produce the clean dataset and reproducible splits that all downstream stages depend on.

| Step | What happens | Output |
|------|-------------|--------|
| Exploration | Count images per class, check dimensions, visualize samples | Understanding of raw data |
| Loading comparison | Load all images into RAM vs. load paths only — measure time and memory | Justification for generator approach |
| Preprocessing | Resize to 64×64, normalize to [0,1], save clean copies | `data/processed/` |
| Split creation | Divide into train / val / test, save as CSVs pointing to processed paths | `data/splits/*.csv` |
| Keras pipeline | `image_dataset_from_directory` reading processed images + augmentation + `.cache()` + `.prefetch()` | Reusable Keras data pipeline |
| PyTorch pipeline | `ImageFolder` reading processed images + `transforms.Compose` + `DataLoader` | Reusable PyTorch data pipeline |

**Key files:** `memory_loader.py`, `generator_loader.py`, `augmentation_keras.py`, `augmentation_torch.py`, `geo_dataloader.py`

---

### Stage 2 — CNN Development
**Goal:** Design, train, and evaluate convolutional neural network models for land cover classification.

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Load split CSVs, read preprocessed images from `data/processed/` | `data/splits/`, `data/processed/` | Batched train/val/test sets |
| Augmentation | Random flips, rotations applied on-the-fly each epoch | Preprocessed image batches | Augmented training batches |
| Model training | Custom CNN trained in Keras and PyTorch | Augmented batches | Trained model weights |
| Evaluation | Accuracy, Precision, Recall, F1-Score on held-out test set | Test split + trained weights | Metrics for Stage 4 |

**Key files:** `cnn_keras.py`, `cnn_pytorch.py`, `evaluate.py`

---

### Stage 3 — Vision Transformer Integration
**Goal:** Apply transfer learning with pre-trained Vision Transformers and benchmark against the CNN baseline from Stage 2.

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Same split CSVs and `data/processed/` as Stage 2 — identical pipeline | `data/splits/`, `data/processed/` | Batched train/val/test sets |
| Fine-tuning | Pre-trained ViT adapted to satellite imagery in Keras and PyTorch | Augmented batches | Trained ViT weights |
| Evaluation | Same metrics as Stage 2 on the same test split — ensures fair comparison | Test split + trained weights | Metrics for Stage 4 |

**Key files:** `vit_keras.py`, `vit_pytorch.py`, `transfer_learning.py`

---

### Stage 4 — Final Report and Wrap-Up
**Goal:** Consolidate all results, compare all models and frameworks, and produce the final deliverable.

| Step | What happens | Input |
|------|-------------|-------|
| Model loading | Load all saved weights from Stage 2 and Stage 3 | `saved_models/` from both stages |
| Final evaluation | Run all models on the same held-out test split | `data/splits/test.csv` + `data/processed/` |
| Comparison | CNN vs ViT, Keras vs PyTorch — metrics table and plots | Evaluation results |
| Report | Document findings, trade-offs, and recommendations | All above |

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
| Deep Learning | Keras (TensorFlow), PyTorch |
| Data Processing | NumPy, Pandas, Rasterio / GDAL |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter / VS Code |
| Evaluation | Scikit-learn |

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/your-username/land-classification.git
cd land-classification

# Install dependencies
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

---

## 📄 License

This project is for academic and internal use only.
