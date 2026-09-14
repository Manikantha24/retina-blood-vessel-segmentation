import os
import random

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 35
LEARNING_RATE = 1e-4
SEED = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "DRIVE",
    "training",
    "images"
)

TRAIN_MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "DRIVE",
    "training",
    "1st_manual"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "unet_resnet34_tversky_bce35.pth"
)


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("RETINA BLOOD VESSEL SEGMENTATION")
print("==========================================")
print("Device       :", device)
print("Image size   :", IMAGE_SIZE)
print("Batch size   :", BATCH_SIZE)
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Random seed  :", SEED)
print("==========================================")


# ==========================================
# DATASET
# ==========================================

class RetinaDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=IMAGE_SIZE
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.images = sorted(
            os.listdir(image_dir)
        )

        self.masks = sorted(
            os.listdir(mask_dir)
        )

        if len(self.images) != len(self.masks):
            raise ValueError(
                "Number of images and masks do not match."
            )

        if len(self.images) == 0:
            raise ValueError(
                "No training images found."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # ----------------------------------
        # Load image
        # ----------------------------------

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------
        # Load mask
        # ----------------------------------

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.array(mask)

        # ----------------------------------
        # Resize
        # ----------------------------------

        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------
        # Normalize
        # ----------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        mask = mask.astype(
            np.float32
        ) / 255.0

        # ----------------------------------
        # Convert to tensors
        # ----------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ==========================================
# DATA VALIDATION
# ==========================================

if not os.path.exists(TRAIN_IMAGE_DIR):
    raise FileNotFoundError(
        f"Training image directory not found:\n"
        f"{TRAIN_IMAGE_DIR}"
    )

if not os.path.exists(TRAIN_MASK_DIR):
    raise FileNotFoundError(
        f"Training mask directory not found:\n"
        f"{TRAIN_MASK_DIR}"
    )


# ==========================================
# DATA LOADER
# ==========================================

train_dataset = RetinaDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Training batches:", len(train_loader))
print("==========================================")


# ==========================================
# MODEL
# ==========================================

print("Creating U-Net model...")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model = model.to(device)

print("Encoder       : ResNet34")
print("Encoder weights: ImageNet pretrained")
print("Architecture  : U-Net")
print("==========================================")


# ==========================================
# LOSS FUNCTIONS
# ==========================================

tversky_loss = smp.losses.TverskyLoss(
    mode="binary",
    from_logits=True,
    alpha=0.3,
    beta=0.7
)

bce_loss = smp.losses.SoftBCEWithLogitsLoss()


def fine_vessel_loss(
    predictions,
    masks
):
    return (
        tversky_loss(predictions, masks)
        + bce_loss(predictions, masks)
    )


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

print("")
print("Starting training...")
print("------------------------------------------")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = fine_vessel_loss(
            predictions,
            masks
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = (
        epoch_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("------------------------------------------")
print("Training completed successfully.")
print("")
print("Model saved to:")
print(MODEL_PATH)
print("")
print("Training configuration:")
print("  Architecture : U-Net")
print("  Encoder      : ResNet34")
print("  Pretrained   : ImageNet")
print("  Loss         : Tversky + BCE")
print("  Epochs       :", EPOCHS)
print("  Threshold    : 0.6")
print("==========================================")