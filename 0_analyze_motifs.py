import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv
from utils.logger import get_logger

# === LOAD ENVIRONMENT CONFIG ===
load_dotenv()
logger = get_logger("analyze_motifs")

ANALYSIS_FILE = os.getenv("ANALYSIS_FILE")
MOTIF_COLUMN = os.getenv("MOTIF_COLUMN", "Motifs")
REPORTS_DIR = os.getenv("REPORTS_DIR", "motif_analysis_reports")

# Ensure output directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# === HELPER FUNCTIONS ===
def clean_and_split_motifs(value):
    """Normalize and split motif strings into lists."""
    if pd.isna(value):
        return []
    value = str(value)
    for sep in [";", ",", "/"]:
        if sep in value:
            return [x.strip().replace(" ", "_") for x in value.split(sep) if x.strip()]
    return [value.strip().replace(" ", "_")]

# === MAIN ANALYSIS ===
logger.info(f"Starting motif frequency analysis for: {ANALYSIS_FILE}")

# Load the file
try:
    ext = os.path.splitext(ANALYSIS_FILE)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(ANALYSIS_FILE)
    elif ext == ".csv":
        df = pd.read_csv(ANALYSIS_FILE)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx.")
    logger.info(f"Loaded file: {ANALYSIS_FILE} with {len(df)} rows.")
except Exception as e:
    logger.error(f"Failed to load file: {e}")
    raise SystemExit(e)

# Verify the motif column exists
if MOTIF_COLUMN not in df.columns:
    logger.error(f"Column '{MOTIF_COLUMN}' not found in dataset.")
    raise SystemExit(f"Column '{MOTIF_COLUMN}' not found in dataset.")

# Process motifs with progress bar
logger.info("Processing motifs with progress tracking...")
motifs_series = []
for value in tqdm(df[MOTIF_COLUMN].dropna(), desc="Cleaning motifs"):
    motifs_series.append(clean_and_split_motifs(value))

# Flatten the list of lists
logger.info("Flattening motif lists...")
all_motifs = [m for sublist in tqdm(motifs_series, desc="Flattening") for m in sublist]

# Count motif occurrences
logger.info("Counting motif occurrences...")
motif_counts = pd.Series(all_motifs).value_counts().rename_axis("Motif").reset_index(name="Count")

# Log summary statistics
logger.info(f"Unique motifs detected: {len(motif_counts)}")

# === VISUALIZATION ===
plt.figure(figsize=(14, 6))
sns.barplot(data=motif_counts.head(50), x="Motif", y="Count", color="skyblue")
plt.xticks(rotation=75, ha="right")
plt.title("Motif Frequency")
plt.xlabel("Motif")
plt.ylabel("Frequency")
plt.tight_layout()

base_name = os.path.splitext(os.path.basename(ANALYSIS_FILE))[0]

chart_path = os.path.join(REPORTS_DIR, f"{base_name}_analysis.png")
plt.savefig(chart_path)
plt.close()
logger.info(f"Saved motif frequency chart to {chart_path}")

# === SAVE REPORT ===
report_path = os.path.join(REPORTS_DIR, f"{base_name}_analysis.csv")
motif_counts.to_csv(report_path, index=False)
logger.info(f"Saved motif frequency report to {report_path}")
