# Haring Motif Classifier  
*A PyTorch-based image classification pipeline for identifying motifs in Keith Haring artworks.*

---

## Overview
This project trains a **ResNet-18** model to automatically identify **visual motifs** in Keith Haring’s artworks (e.g., *Radiant Baby*, *Barking Dog*, *Dancing Figures*, etc.).  
It uses a labeled CSV dataset of artworks, downloads corresponding images, splits them into training and validation sets, and trains a **multi-label image classifier** — since each image can contain more than one motif.

The system is organized into three main Python scripts:

| Script | Purpose |
|--------|----------|
| **1_create_dataset.py** | Downloads images, splits data, and builds label mappings. |
| **2_train.py** | Trains the model using PyTorch and ResNet-18. |
| **3_predict.py** | Runs inference on new images or entire folders, outputs motif predictions. |
| **0_analyze_motifs.py** | Analyzes motif frequencies in a csv file, visualizes distributions. |

Logging and configuration are handled through environment variables and reusable utilities.

---

## Project Structure

```
haring-motif-classifier/
├── 0_analyze_motifs.py
├── 1_create_dataset.py
├── 2_train.py
├── 3_predict.py
├── utils/
│   └── logger.py
├── data/
│   ├── data.csv
│   ├── train/
│   ├── val/
│   └── *.json
├── logs/
│   └── ...
├── prediction_results/
├── motif_analysis_reports/
├── .env
├── .gitignore
└── README.md
```

---

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1 Clone the Repository
```bash
git clone https://github.com/SelinaMangaroo/haring-motif-classifier.git
cd haring-motif-classifier
```

### 2 Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

### 3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4 Prepare the `.env` File
Add and edit a `.env` file to match your paths and settings (see configuration below).
Copy and adjust environment variables as needed.

### 5 Add Dataset CSV
Place your labeled CSV (e.g. `*-data.csv`) inside the `data/` directory.
Each file must include:
- A column with image URLs (e.g. Medium resolution media)
- A column with motif labels (e.g. Motifs)

### 6 Run the Scripts
```bash
python 1_create_dataset.py
python 2_train.py
python 3_predict.py
```

### 7 (Optional) Analyze Motif Distribution
```bash
python 0_analyze_motifs.py
```

All scripts will automatically generate logs, models, and output JSONs in timestamped folders.

---


## How It Works

### 1. Dataset Creation (`1_create_dataset.py`)
- Reads the CSV file (e.g., `haring-data.csv`).
- Splits into **train** and **validation** sets (default: 80/20).
- Downloads each image from its URL and stores it in:
  ```
  data/train/images/
  data/val/images/
  ```
- Builds two JSON files mapping filenames → motif lists:
  - `train_labels.json`
  - `val_labels.json`
- Automatically logs all operations to timestamped log files in `logs/`.

### 2. Model Training (`2_train.py`)
- Loads pre-trained **ResNet-18** from ImageNet.
- Freezes early convolutional layers and retrains only the final fully connected layer.
- Outputs:
  - Trained model weights → `haring_resnet18_model.pth`
  - Training/validation loss curves → `training_loss_curve.png`
  - Logs progress and loss metrics per epoch.

### 3. Prediction (`3_predict.py`)
- Loads the trained model and runs inference on:
  - A **single image**, or an **entire directory** of images.
- Filters motifs above a confidence threshold (default: `0.5`).
- Outputs a JSON file named like:
  ```
  predictions_11_13_2025_16_42_10.json
  ```
  containing results in this format:
  ```json
  {
    "PS-0012.jpg": {
      "Radiant_Baby": 0.91,
      "Social_Activism–Art": 0.67
    }
  }
  ```

### 4. Motif Analysis (`0_analyze_motifs.py`)
- Loads any CSV or Excel file and counts motif frequencies.
- Detects rare or over-represented motifs (outlier detection).
Generates:
- `*_analysis.csv`
`*_analysis.png` (bar chart of top motifs)

---

## Environment Configuration (`.env`)
All runtime settings are managed in a `.env` file:

```
# === Dataset Configuration ===
CSV_PATH=data/data.csv
DATA_DIR=data
TRAIN_SPLIT=0.8
IMAGE_COLUMN=Medium resolution media
MOTIF_COLUMN=Motifs

# === Logging ===
LOG_DIR=logs

# === Training Configuration ===
BATCH_SIZE=16
EPOCHS=6
LEARNING_RATE=0.0001
MODEL_PATH=resnet18_model.pth

# === Prediction Configuration ===
THRESHOLD=0.5
PREDICTION_PATH=prediction-images
SAVE_PREDICTIONS_PATH=prediction_results

# === MOTIF ANALYSIS CONFIGURATION ===
ANALYSIS_FILE=data/haring_csv_data/export_set.csv
MOTIF_COLUMN=Motifs
REPORTS_DIR=motif_analysis_reports
```

| **Category**   | **Variable**      | **Description**                                                             | **Example / Default Value**         |
| -------------- | ----------------- | --------------------------------------------------------------------------- | ----------------------------------- |
| **Dataset**    | `CSV_PATH`        | Path to the input CSV file containing metadata and image URLs.              | `data/haring-data.csv`              |
|                | `DATA_DIR`        | Root directory where dataset folders (train/val) and labels will be stored. | `data`                              |
|                | `TRAIN_SPLIT`     | Percentage of data to use for training (remainder used for validation).     | `0.8`                               |
|                | `IMAGE_COLUMN`    | Column name in CSV containing image URLs.                                   | `Medium resolution media`           |
|                | `MOTIF_COLUMN`    | Column name in CSV containing motif labels.                                 | `Motifs`                            |
| **Training**   | `BATCH_SIZE`      | Number of images per training batch.                                        | `16`                                |
|                | `EPOCHS`          | Number of epochs (full passes through the dataset).                         | `6`                                 |
|                | `LEARNING_RATE`   | Learning rate for optimizer — smaller = slower but more stable.             | `0.0001`                            |
|                | `MODEL_PATH`      | Path where the trained model will be saved.                                 | `haring_resnet18_model.pth`         |
| **Prediction** | `THRESHOLD`       | Confidence threshold (0–1) for including a motif in predictions.            | `0.5`                               |
|                | `PREDICTION_PATH` | Path to image or folder for running predictions.                            | `prediction-imgs/15098_small.jpg`   |
|                | `SAVE_PREDICTIONS_PATH` | Path to save prediction results (Excel file and JSON)                 | `prediction_results`                |
| **Analysis**   | `ANALYSIS_FILE`   | Path to CSV or Excel file to analyze.                                       | `export_set.csv`                    |
|                | `MOTIF_COLUMN`    | Column name containing motifs.                                              | `Motifs`                            |
|                | `REPORTS_DIR`     | Directory where reports will be saved.                                      | `motif_analysis_reports`            |
| **Logging**    | `LOG_DIR`         | Directory where log files will be stored.                                   | `logs`                              |


You can adjust these values to control training speed, dataset location, or prediction thresholds.

---

## Logging
All processes generate detailed timestamped log files (under `logs/`) via `utils/logger.py`, e.g.:

```
logs/
├── create_dataset_11_13_2025_15_45_40.log
├── train_11_13_2025_15_54_11.log
└── predict_11_13_2025_16_22_33.log
```

Each log records start/end times, errors, and progress updates.

---

## GPU acceleration
If you have CUDA installed, the script will automatically use your GPU:
```
Using device: cuda
```

---

## 👩‍💻 **Author**

**Selina Mangaroo**  
*Software Engineer*

🌐 [Website](https://www.selinamangaroo.com/)  
💼 [LinkedIn](https://www.linkedin.com/in/selinamangaroo/)  
📦 [GitHub](https://github.com/SelinaMangaroo)  
✍️ [Medium](https://medium.com/@selinamangaroo)

---