# 🌍 Agricultural Land Classification System

> A deep learning pipeline for satellite imagery classification using CNNs and Vision Transformers — built with Keras and PyTorch.

---

## 📌 Project Overview

This capstone project develops a **multi-class terrain classification system** for agricultural applications. Using satellite imagery, the system identifies and classifies different land cover types such as:

- 🌾 Croplands
- 🌲 Forests
- 💧 Water Bodies
- 🏙️ Urban Areas
- 🪨 Barren Land

The project runs **parallel implementations in Keras and PyTorch**, enabling direct framework comparison across all stages.

---

## 🗂️ Project Structure

```
land-classification/
│
├── data/
│   ├── raw/                  # Original satellite images
│   ├── processed/            # Preprocessed and augmented data
│   └── splits/               # Train / Validation / Test splits
│
├── module1_data/
│   ├── memory_loader.py      # Memory-based data loading
│   ├── generator_loader.py   # Generator-based data loading
│   ├── augmentation_keras.py # Keras augmentation pipeline
│   ├── augmentation_torch.py # PyTorch augmentation pipeline
│   └── geo_dataloader.py     # Custom geospatial data loader
│
├── module2_cnn/
│   ├── cnn_keras.py          # CNN model in Keras
│   ├── cnn_pytorch.py        # CNN model in PyTorch
│   └── evaluate.py           # Accuracy, Precision, Recall evaluation
│
├── module3_transformer/
│   ├── vit_keras.py          # Vision Transformer (Keras)
│   ├── vit_pytorch.py        # Vision Transformer (PyTorch)
│   └── transfer_learning.py  # Fine-tuning pre-trained models
│
├── module4_final/
│   ├── comparison.py         # CNN vs ViT comparative analysis
│   └── report/               # Final project report
│
├── notebooks/                # Jupyter notebooks per module
├── requirements.txt
└── README.md
```

---

## 🧩 Modules

### Module 1 — Data Handling
**Goal:** Efficient data loading and augmentation for geospatial image datasets.

| Task | Description |
|------|-------------|
| Memory vs Generator Loading | Compare in-memory and lazy loading strategies |
| Data Augmentation | Flipping, rotation, color jitter, cropping |
| Custom Geo DataLoader | Tailored loader for satellite image formats |

---

### Module 2 — CNN Development
**Goal:** Design and train CNN models for land classification.

| Task | Description |
|------|-------------|
| Keras CNN | Custom CNN architecture in Keras |
| PyTorch CNN | Equivalent CNN architecture in PyTorch |
| Evaluation | Accuracy, Precision, Recall, F1-Score |

---

### Module 3 — Vision Transformer Integration
**Goal:** Apply transfer learning with Vision Transformers.

| Task | Description |
|------|-------------|
| Fine-tuning (Keras) | Pre-trained ViT fine-tuned on terrain dataset |
| Fine-tuning (PyTorch) | Pre-trained ViT fine-tuned on terrain dataset |
| CNN vs ViT Comparison | Performance benchmarking across both model families |

---

### Module 4 — Final Report & Wrap-Up
**Goal:** Consolidate findings and submit final deliverables.

- Comparative analysis of all models
- Metrics summary: Accuracy, F1-Score, AU-ROC
- Insights on framework trade-offs (Keras vs PyTorch)
- Final project report

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
| Notebooks | Jupyter / Google Colab |
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