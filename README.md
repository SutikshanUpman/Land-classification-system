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

## Project Structure

```
land-classification/
├── data/
│   ├── raw/              # Original satellite images
│   └── splits/           # train.csv / val.csv / test.csv
├── stage1_data/          # Data loading & augmentation
├── stage2_cnn/           # CNN models (Keras + PyTorch)
├── stage3_transformer/   # Vision Transformer models
├── stage4_final/         # Final evaluation & report
├── notebooks/            # Jupyter notebooks
└── requirements.txt
```

---

## Tech Stack

- **Frameworks:** TensorFlow/Keras 2.20.0, PyTorch
- **Tools:** NumPy, Pandas, OpenCV, Scikit-learn
- **Environment:** Python 3.10.19

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

## Setup

```bash
git clone https://github.com/your-username/land-classification.git
cd land-classification
pip install -r requirements.txt
```

---

## Usage

1. Run notebooks in `notebooks/` sequentially (01 → 04)
2. Train CNN models (Stage 2)
3. Fine-tune ViT models (Stage 3)
4. Generate final report (Stage 4)

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
