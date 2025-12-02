import os, json, torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import ssl, certifi
from dotenv import load_dotenv
from utils.logger import get_logger

# Fix for SSL certificate issues when downloading pretrained models
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# === LOAD CONFIG ===
load_dotenv()
logger = get_logger("train")

DATA_DIR = os.getenv("DATA_DIR", "data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
EPOCHS = int(os.getenv("EPOCHS", 6))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
logger.info(f"DATA_DIR={DATA_DIR}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LEARNING_RATE}")

# # === LOAD LABEL FILES ===
# # Load train and validation label mappings created earlier by create_dataset.py
# try:
#     with open(os.path.join(DATA_DIR, "train_labels.json")) as f:
#         train_labels = json.load(f)
#     with open(os.path.join(DATA_DIR, "val_labels.json")) as f:
#         val_labels = json.load(f)
#     logger.info("Successfully loaded label mappings.")
# except Exception as e:
#     logger.error(f"Error loading label JSONs: {e}")
#     raise SystemExit(e)


# === LOAD ALL LABEL FILES ===
def load_all_labels(prefix):
    """Load and merge all JSON label files matching the prefix (train or val)."""
    all_labels = {}
    for fname in os.listdir(DATA_DIR):
        if fname.startswith(prefix) and fname.endswith(".json"):
            path = os.path.join(DATA_DIR, fname)
            try:
                with open(path) as f:
                    labels = json.load(f)
                    all_labels.update(labels)
                logger.info(f"Loaded {len(labels)} entries from {fname}")
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")
    return all_labels

train_labels = load_all_labels("train_labels")
val_labels = load_all_labels("val_labels")

logger.info(f"Total combined: {len(train_labels)} train / {len(val_labels)} val samples.")

# === PREPARE MOTIF LIST ===
# Extract all unique motifs from both train and validation sets for consistent indexing
all_motifs = sorted({m for labels in [train_labels, val_labels] for v in labels.values() for m in v})
motif_to_idx = {m: i for i, m in enumerate(all_motifs)}
logger.info(f"Detected {len(all_motifs)} motif classes.")

# === CUSTOM DATASET ===
class HaringDataset(Dataset):
    
    """
    Custom PyTorch Dataset for Haring motif classification.
    Handles loading images and converting motif lists into binary label vectors.
    
    Args:
        img_dir (str): Directory containing images.
        label_dict (dict): Mapping of filenames to lists of motifs.
        motifs (list): Master list of all motifs (used for indexing).
        transform (callable, optional): Optional transformations applied to each image.
    """
    def __init__(self, img_dir, label_dict, motifs, transform=None):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.motifs = motifs
        self.transform = transform
        self.filenames = list(label_dict.keys())

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieve one image and its label vector.
        Args:
            idx (int): Index of the item.
        Returns:
            image (Tensor): The preprocessed image.
            label_vec (Tensor): encoded vector (1 for present motifs).
        """
        fname = self.filenames[idx]
        path = os.path.join(self.img_dir, fname)
        image = Image.open(path).convert("RGB")

        # Initialize label vector (length = number of motif classes)
        label_vec = torch.zeros(len(self.motifs))
        for motif in self.label_dict[fname]:
            if motif in self.motifs:
                label_vec[self.motifs.index(motif)] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, label_vec

# === IMAGE TRANSFORMS ===
# Augmentation adds variety and prevents overfitting; normalization matches ResNet training stats
train_transforms = transforms.Compose([
    transforms.Resize((224,224)), # Resize to 224×224 (ResNet input size)
    transforms.RandomHorizontalFlip(), # Randomly flip horizontally
    transforms.RandomRotation(10), # Random rotation 10 degrees
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize using ImageNet mean/std
])
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# === DATA LOADERS ===
# Wrap datasets in DataLoaders to handle batching, shuffling, and parallel loading
train_dataset = HaringDataset(os.path.join(DATA_DIR, "train/images"), train_labels, all_motifs, train_transforms)
val_dataset   = HaringDataset(os.path.join(DATA_DIR, "val/images"), val_labels, all_motifs, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# === MODEL SETUP ===

# Load pretrained ResNet18 and fine-tune its final layer for multi-label classification
logger.info("Loading pretrained ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all earlier layers to keep pretrained visual features intact
for p in model.parameters():
    p.requires_grad = False 

# Replace final fully connected layer with new output layer for motifs
model.fc = nn.Linear(model.fc.in_features, len(all_motifs))
model = model.to(DEVICE) # move model to GPU if available

# Define loss and optimizer
# BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid (best for multi-label)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
train_losses, val_losses = [], []
logger.info("Starting training...")

for epoch in range(EPOCHS):
    
    # ---- TRAINING PHASE ----
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad() # Reset gradients
        outputs = model(imgs) # Forward pass (predictions)
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backpropagate errors
        optimizer.step() # Update weights
        running_loss += loss.item() # accumulate loss

    avg_train_loss = running_loss / len(train_loader)

    # ---- VALIDATION PHASE ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient computation for efficiency
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    logger.info(f"Epoch {epoch+1}/{EPOCHS}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

# === SAVE MODEL ===
# Save trained model weights and motif index mapping for use during prediction
model_path = os.getenv("MODEL_PATH", "haring_resnet18_model.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "motif_to_idx": motif_to_idx
}, "haring_resnet18_model.pth")
logger.info(f"Model saved to {model_path}")

# === PLOT LOSSES ===
# Visualize training progress
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend(); 
plt.xlabel("Epoch"); 
plt.ylabel("Loss"); 
plt.title("Training vs Validation Loss")
plt.savefig("training_loss_curve.png")
plt.close()
logger.info("Saved training_loss_curve.png and completed training successfully.")