# 🌍 Agricultural Land Classification System

> A deep learning pipeline for satellite imagery binary classification using CNNs and Vision Transformers — built with Keras (TensorFlow 2.20.0) and PyTorch (Python 3.10.19).

---

## 📌 Project Overview

This capstone project develops a **binary land classification system** for agricultural applications. Using Sentinel-2 satellite imagery, the system classifies land into two categories:

- 🌾 **class_1_agri** — Cultivable / agricultural land  
- 🪨 **class_0_non_agri** — Non-cultivable / non-agricultural land

**Dataset:** 6,000 images total (3,000 per class), Sentinel-2 satellite tiles.  
**Split:** 70% train / 15% val / 15% test → 4,200 / 900 / 900 images (stratified, `random_state=42`).

The project runs **parallel implementations in Keras and PyTorch**, enabling direct framework comparison across all stages.

---

## 🎯 Success Criteria

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 90% on held-out test set |
| F1-Score | ≥ 0.90 (macro average) |
| AU-ROC | ≥ 0.95 |

These targets are based on prior binary land-use classification benchmarks on Sentinel-2 imagery at comparable resolution.

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
│   ├── 01_memory_vs_path_loading.ipynb
│   ├── 02_keras_custom_generator.ipynb
│   ├── 03_keras_utility_pipeline.ipynb
│   └── 04_pytorch_custom_dataset.ipynb
│
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
  ✔ Resize all images to 64×64 using PIL
  ✔ Convert to RGB
  ✔ Skip corrupt images
  ✖ Do NOT normalize pixel values

Stage 2 & Stage 3 — Training-time (live, per epoch):
  ✔ Normalize pixel values to [0, 1] (Rescaling / ToTensor)
  ✔ Random augmentation:
      - Horizontal / vertical flip
      - Random rotation (±10–15°)
      - Random zoom (Keras only: 0.1)
```

---

### 🔒 Data Consistency Rule

All stages MUST load from the same CSV splits:

```
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

Split creation uses `sklearn.model_selection.train_test_split` with `stratify=df["label"]` and `random_state=42`.

❌ Do NOT use framework-based splitting (e.g., `validation_split` in Keras) for training.  
`image_dataset_from_directory` with `validation_split` is used in Stage 1 (Notebook 3) **for demonstration only** to compare Keras built-in loading vs. the custom CSV-based approach. All training in Stages 2 & 3 must use the CSV splits.

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
│     - Memory (37.94 MB, 2.95s) vs           │
│       generator (0.54 MB, 0.03s) comparison │
│                                             │
│  2. Preprocess and save to disk (once)      │
│     - Resize all images to 64×64            │
│     - Save to data/processed/               │
│                                             │
│  3. Create reproducible splits              │
│     - Stratified 70/15/15, random_state=42  │
│     - Save CSVs (paths → data/splits/)      │
│                                             │
│  4. Build and validate augmentation pipes   │
│     - Keras: tf.data + augmentation layers  │
│     - PyTorch: Custom Dataset + transforms  │
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

## ⚠️ Final Data Pipeline Decision

- The **CSV-based custom generator** is the official training pipeline.
- It ensures reproducibility and consistency across Keras & PyTorch.

- `image_dataset_from_directory` is used **only for experimentation** (Notebook 3: augmentation understanding, tf.data API) and NOT used in final training.

---

## 🧩 Pipeline Stages

### Stage 1 — Data Handling

**Notebooks:** `01_memory_vs_path_loading.ipynb`, `02_keras_custom_generator.ipynb`, `03_keras_utility_pipeline.ipynb`, `04_pytorch_custom_dataset.ipynb`

| Step | What happens | Output |
|------|-------------|--------|
| Exploration | Visualize samples, check class balance | Understanding of raw data |
| Loading comparison | RAM-based (37.94 MB, 2.95s) vs path-based (0.54 MB, 0.03s) | Justification for generator approach |
| Preprocessing | Resize to 64×64 RGB via PIL, skip corrupt images | `data/processed/` |
| Split creation | 70/15/15 stratified split (`random_state=42`), saved as CSVs | `data/splits/*.csv` |
| Keras pipeline | Custom generator + `image_dataset_from_directory` (demo) + augmentation + `.cache().prefetch()` | Keras data pipeline |
| PyTorch pipeline | Custom `LandDataset(Dataset)` + `transforms.Compose` + `DataLoader` | PyTorch data pipeline |

**Key files:** `memory_loader.py`, `generator_loader.py`, `augmentation_keras.py`, `augmentation_torch.py`

#### Stage 1 Implementation Details

**Keras custom generator** (`02_keras_custom_generator.ipynb`):
- Reads `filepath` and `label` columns from CSVs
- Shuffles indices with `np.random.shuffle` at the start of each epoch
- Loads images via `cv2.imread` → BGR→RGB conversion → normalize to `[0, 1]`
- Augmentation: `cv2.flip` (horizontal, p=0.5) + `cv2.warpAffine` rotation (±15°, p=0.5)
- Yields `(batch_images, batch_labels)` as NumPy arrays; loops indefinitely

**Keras utility pipeline** (`03_keras_utility_pipeline.ipynb`):
- `image_dataset_from_directory` with `validation_split=0.2`, `seed=1337` — demo only
- `tf.keras.layers.Rescaling(1/255)` for normalization
- `tf.keras.Sequential` augmentation: `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.2)`, `RandomZoom(0.1)`
- Applied via `.map(num_parallel_calls=AUTOTUNE)` → `.cache()` → `.prefetch(AUTOTUNE)`

**PyTorch custom dataset** (`04_pytorch_custom_dataset.ipynb`):
- `LandDataset(Dataset)` — reads filepath/label from DataFrame, loads via `cv2.imread`, resizes to `(64, 64)`, converts BGR→RGB
- Train transforms: `ToPILImage → RandomHorizontalFlip → RandomRotation(10) → ToTensor` (auto-scales to `[0, 1]`)
- Val/test transforms: `ToPILImage → ToTensor` only
- `DataLoader(batch_size=32, shuffle=True)` for train; `shuffle=False` for val/test

---

### Stage 2 — CNN Development

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Load split CSVs, read from `data/processed/` | `data/splits/`, `data/processed/` | Batched train/val/test |
| Augmentation | Random flips, rotations — live each epoch | Preprocessed batches | Augmented training batches |
| Model training | Custom CNN in Keras and PyTorch | Augmented batches | Trained weights |
| Evaluation | Accuracy, Precision, Recall, F1, AU-ROC on test set | Test split + weights | Metrics for Stage 4 |

**Key files:** `cnn_keras.py`, `cnn_pytorch.py`, `evaluate.py`

#### CNN Architecture (planned)

| Layer | Config |
|-------|--------|
| Conv2D block 1 | 32 filters, 3×3, ReLU + MaxPool 2×2 |
| Conv2D block 2 | 64 filters, 3×3, ReLU + MaxPool 2×2 |
| Conv2D block 3 | 128 filters, 3×3, ReLU + MaxPool 2×2 |
| Flatten | — |
| Dense | 256, ReLU + Dropout 0.5 |
| Output | 1 unit, Sigmoid (binary) |

**Training config:**

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 30 (early stopping, patience=5) |
| Loss | Binary crossentropy |
| Input shape | (64, 64, 3) |
| Random seed | 42 |

---

### Stage 3 — Vision Transformer Integration

| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| Data loading | Same CSV splits and `data/processed/` as Stage 2 | `data/splits/`, `data/processed/` | Batched train/val/test |
| Fine-tuning | Pre-trained ViT adapted in Keras and PyTorch | Augmented batches | Trained ViT weights |
| Evaluation | Same metrics on same test split as Stage 2 | Test split + weights | Metrics for Stage 4 |

**Key files:** `vit_keras.py`, `vit_pytorch.py`, `transfer_learning.py`

#### ViT Architecture & Transfer Learning

**Pretrained checkpoint:** `google/vit-base-patch16-224` (ImageNet-21k pretrained, available via HuggingFace `transformers`)

**Adaptation strategy:**
- Input images upsampled from 64×64 to 224×224 at load time to match ViT patch expectations
- Freeze all transformer encoder layers initially; fine-tune the classification head for 10 epochs
- Unfreeze the last 2 transformer blocks and fine-tune end-to-end for up to 20 additional epochs

**Training config:**

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate (head) | 1e-3 |
| Learning rate (fine-tune) | 1e-5 |
| Batch size | 32 |
| Patch size | 16×16 |
| Input size | 224×224 |
| Random seed | 42 |

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

## 🔁 Reproducibility

| Factor | Value |
|--------|-------|
| Python version | 3.10.19 |
| TensorFlow / Keras | 2.20.0 |
| PyTorch | see `requirements.txt` |
| Sklearn split seed | `random_state=42` |
| Keras utility seed | `seed=1337` (demo only) |
| PyTorch DataLoader | `shuffle=True` (train), `False` (val/test) |
| Global random seed | Set at notebook start via `np.random.seed(42)`, `torch.manual_seed(42)`, `tf.random.set_seed(42)` |

All CSV splits are generated once in Notebook 2 and reused by all subsequent stages. Re-running Notebook 2 will regenerate identical splits due to the fixed `random_state=42`.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | Keras / TensorFlow 2.20.0, PyTorch |
| Data Processing | NumPy, Pandas, Pillow, OpenCV (`cv2`) |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter / VS Code (kernel: `ai-gpu`, Python 3.10.19) |
| Evaluation | Scikit-learn |
| Progress | tqdm |

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/land-classification.git
cd land-classification
pip install -r requirements.txt
```

**Run order:**

```
Notebook 01  →  understand loading strategies (no output written)
Notebook 02  →  preprocess images + generate CSV splits  ← must run first
Notebook 03  →  explore Keras tf.data pipeline (demo only)
Notebook 04  →  verify PyTorch pipeline matches Keras output
Stage 2      →  train CNN (Keras + PyTorch)
Stage 3      →  fine-tune ViT (Keras + PyTorch)
Stage 4      →  evaluate all models + produce report
```

---

## 📈 Results Summary *(to be updated after Stage 4)*

| Model | Framework | Accuracy | F1-Score | AU-ROC |
|-------|-----------|----------|----------|--------|
| CNN | Keras | — | — | — |
| CNN | PyTorch | — | — | — |
| ViT (fine-tuned) | Keras | — | — | — |
| ViT (fine-tuned) | PyTorch | — | — | — |

---

## 👤 Author

**AI Engineer** — Fertilizer Company Land Intelligence Team  
*Capstone Project — Deep Learning for Agricultural Applications*

## 📄 License

This project is for academic and internal use only.
