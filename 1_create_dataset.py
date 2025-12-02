# import os, json, requests, re, pandas as pd
# from tqdm import tqdm
# from PIL import Image
# from io import BytesIO
# from utils.logger import get_logger
# from dotenv import load_dotenv

# logger = get_logger("create_dataset")
# logger.info("Starting dataset creation process")

# load_dotenv()
# CSV_PATH = os.getenv("CSV_PATH", "")
# DATA_DIR = os.getenv("DATA_DIR", "data")
# TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.8))
# IMAGE_COLUMN = os.getenv("IMAGE_COLUMN", "Medium resolution media")
# MOTIF_COLUMN = os.getenv("MOTIF_COLUMN", "Motifs")

# # Create necessary folder structure (train/val images)
# os.makedirs(f"{DATA_DIR}/train/images", exist_ok=True)
# os.makedirs(f"{DATA_DIR}/val/images", exist_ok=True)
# logger.info(f"Ensured dataset folders exist under {DATA_DIR}/")

# # === LOAD AND CLEAN DATA ===
# # Read the CSV file and drop any rows missing image URLs or motif tags
# try:
#     df = pd.read_csv(CSV_PATH).dropna(subset=[IMAGE_COLUMN, MOTIF_COLUMN])
#     logger.info(f"Loaded CSV from {CSV_PATH} with {len(df)} rows.")
# except Exception as e:
#     logger.error(f"Failed to read CSV: {e}")
#     raise SystemExit(e)

# def split_motifs(m):
#     """
#     Split motif string into a clean list of motifs.
#     Handles separators like semicolons, commas, and slashes.
#         Args:
#             m (str): Raw motif string from CSV.
#         Returns:
#             list: Cleaned list of motif strings.
#     """
#     for sep in [";", ",", "/"]:
#         if sep in str(m):
#             return [x.strip().replace(" ", "_") for x in str(m).split(sep) if x.strip()]
#     # if no separator found, return single-element list
#     return [str(m).strip().replace(" ", "_")]

# # Apply motif splitting to each row and store result in new column
# df["motif_list"] = df[MOTIF_COLUMN].apply(split_motifs)
# logger.info("Split motifs into list format for all rows.")

# # Randomly shuffle the dataset to ensure even distribution before splitting
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Split dataset into training and validation subsets
# cut = int(len(df)*TRAIN_SPLIT)
# train_df, val_df = df.iloc[:cut], df.iloc[cut:]
# logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation rows.")

# def safe_filename(name):
#     """
#     Convert a text string into a safe filename by replacing problematic characters.
#     Args:
#         name (str): Original string (e.g., object identifier).
#     Returns:
#         str: Safe filename string.
#     Example:
#         "KHA-0491/17 04/17" → "KHA-0491_17_04_17"
#     """
#     return re.sub(r"[^\w\-_.]", "_", str(name))

# def download_and_save(row, split):
#     """
#     Download a single image from its URL and save it to disk.
    
#     Args:
#         row (pd.Series): A single row from the dataframe.
#         split (str): Either 'train' or 'val' (determines output folder).
    
#     Returns:
#         str or None: Saved filename if successful, otherwise None.
#     """
#     url = row[IMAGE_COLUMN]
#     try:
#         # Fetch image from URL
#         r = requests.get(url, timeout=10)
#         r.raise_for_status() # Raise error if request fails
#         img = Image.open(BytesIO(r.content)).convert("RGB") # Open as RGB image
#     except Exception as e:
#         logger.warning(f"Failed to download {url}: {e}")
#         return None

#     # Create a safe filename based on object identifier
#     fname = f"{safe_filename(row['Object identifier'])}.jpg"
#     path = os.path.join(DATA_DIR, split, "images", fname)
#     os.makedirs(os.path.dirname(path), exist_ok=True)
    
#     try:
#         img.save(path)
#     except Exception as e:
#         logger.warning(f"Could not save {fname}: {e}")
#         return None
    
#     logger.info(f"Saved image {fname} ({split})")
#     return fname

# def build_split(df_split, split):
#     """
#     Download all images for a given data split and create a JSON label map.
    
#     For each row:
#     - Downloads the image
#     - Maps the saved filename to its list of motifs
#     - Stores results in {split}_labels.json
#     Args:
#         df_split (pd.DataFrame): DataFrame for the data split (train or val).
#         split (str): Either 'train' or 'val'.
#     Returns:
#         None
#     """
#     mapping = {}
#     logger.info(f"Building {split} dataset... ({len(df_split)} images)")

#     for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
#         fname = download_and_save(row, split)
#         if fname:
#             mapping[fname] = row["motif_list"]
    
#     # Save mapping as JSON for training
#     json_path = f"{DATA_DIR}/{split}_labels.json"
#     with open(json_path, "w") as f:
#         json.dump(mapping, f, indent=2)
#     logger.info(f"Saved {json_path} with {len(mapping)} labeled entries.")

# # === EXECUTION ===
# # Build both training and validation sets
# build_split(train_df, "train")
# build_split(val_df, "val")

# logger.info("Dataset creation process completed successfully.")


import os, json, requests, re, pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from utils.logger import get_logger
from dotenv import load_dotenv

# === INITIALIZE LOGGER AND ENVIRONMENT ===
logger = get_logger("create_dataset")
logger.info("Starting dataset creation process")

load_dotenv()
CSV_PATH = os.getenv("CSV_PATH", "data/haring-data.csv")
DATA_DIR = os.getenv("DATA_DIR", "data")
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.8))
IMAGE_COLUMN = os.getenv("IMAGE_COLUMN", "Medium resolution media")
MOTIF_COLUMN = os.getenv("MOTIF_COLUMN", "Motifs")

# Create necessary folder structure (train/val images)
os.makedirs(f"{DATA_DIR}/train/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/val/images", exist_ok=True)
logger.info(f"Ensured dataset folders exist under {DATA_DIR}/")

# === HELPER FUNCTIONS ===
def split_motifs(m):
    """Split motif string into a clean list of motifs."""
    for sep in [";", ",", "/"]:
        if sep in str(m):
            return [x.strip().replace(" ", "_") for x in str(m).split(sep) if x.strip()]
    return [str(m).strip().replace(" ", "_")]

def safe_filename(name):
    """Convert a text string into a filesystem-safe filename."""
    return re.sub(r"[^\w\-_.]", "_", str(name))

def download_and_save(row, split):
    """Download a single image and save it to disk."""
    url = row[IMAGE_COLUMN]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None

    fname = f"{safe_filename(row.get('Object identifier', 'unknown'))}.jpg"
    path = os.path.join(DATA_DIR, split, "images", fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        img.save(path)
    except Exception as e:
        logger.warning(f"Could not save {fname}: {e}")
        return None

    logger.info(f"Saved image {fname} ({split})")
    return fname

def build_split(df_split, split, label_suffix=""):
    """Download all images for a given data split and create a JSON label map."""
    mapping = {}
    logger.info(f"Building {split} dataset... ({len(df_split)} images)")

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"{split}"):
        fname = download_and_save(row, split)
        if fname:
            mapping[fname] = row["motif_list"]

    json_path = f"{DATA_DIR}/{split}_labels{label_suffix}.json"
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved {json_path} with {len(mapping)} labeled entries.")

def process_csv(csv_path):
    """Load, clean, split, and build dataset for a single CSV."""
    try:
        df = pd.read_csv(csv_path).dropna(subset=[IMAGE_COLUMN, MOTIF_COLUMN])
        logger.info(f"Loaded CSV: {csv_path} with {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    df["motif_list"] = df[MOTIF_COLUMN].apply(split_motifs)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    cut = int(len(df) * TRAIN_SPLIT)
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]
    logger.info(f"Split {os.path.basename(csv_path)} into {len(train_df)} train / {len(val_df)} val rows.")

    # Use base name of CSV to create unique label files
    suffix = "_" + os.path.splitext(os.path.basename(csv_path))[0]

    build_split(train_df, "train", suffix)
    build_split(val_df, "val", suffix)

# === EXECUTION ===
# Handle single CSV or directory of CSVs
if os.path.isdir(CSV_PATH):
    csv_files = [os.path.join(CSV_PATH, f) for f in os.listdir(CSV_PATH) if f.lower().endswith(".csv")]
    logger.info(f"Detected directory: processing {len(csv_files)} CSV files.")
    for csv_file in csv_files:
        process_csv(csv_file)
else:
    process_csv(CSV_PATH)

logger.info("Dataset creation process completed successfully.")