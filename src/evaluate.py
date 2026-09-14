import os

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.6

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "DRIVE",
    "test",
    "images"
)

TEST_MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "DRIVE",
    "test",
    "1st_manual"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "unet_resnet34_tversky_bce35.pth"
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("RETINA BLOOD VESSEL SEGMENTATION")
print("MODEL EVALUATION")
print("==========================================")
print("Device    :", device)
print("Threshold :", THRESHOLD)
print("==========================================")


# ==========================================
# LOAD MODEL
# ==========================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print("==========================================")


# ==========================================
# METRIC FUNCTIONS
# ==========================================

def calculate_dice(
    prediction,
    target
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    dice = (
        2.0 * intersection
        / (
            prediction.sum()
            + target.sum()
            + 1e-7
        )
    )

    return dice


def calculate_iou(
    prediction,
    target
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    union = np.logical_or(
        prediction,
        target
    ).sum()

    iou = (
        intersection
        / (union + 1e-7)
    )

    return iou


# ==========================================
# TEST DATA
# ==========================================

test_images = sorted(
    os.listdir(TEST_IMAGE_DIR)
)

test_masks = sorted(
    os.listdir(TEST_MASK_DIR)
)

if len(test_images) != len(test_masks):
    raise ValueError(
        "Number of test images and masks do not match."
    )

print("Test images:", len(test_images))
print("==========================================")


# ==========================================
# EVALUATION
# ==========================================

dice_scores = []
iou_scores = []

print("")
print("Evaluating model...")
print("------------------------------------------")

with torch.no_grad():

    for image_name, mask_name in zip(
        test_images,
        test_masks
    ):

        # ----------------------------------
        # Load image
        # ----------------------------------

        image_path = os.path.join(
            TEST_IMAGE_DIR,
            image_name
        )

        image = cv2.imread(
            image_path
        )

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
            TEST_MASK_DIR,
            mask_name
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
            IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------
        # Normalize image
        # ----------------------------------

        image = (
            image.astype(np.float32)
            / 255.0
        )

        # ----------------------------------
        # Convert image to tensor
        # ----------------------------------

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------
        # Model prediction
        # ----------------------------------

        prediction = model(
            image_tensor
        )

        prediction = torch.sigmoid(
            prediction
        )

        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )

        # ----------------------------------
        # Apply threshold
        # ----------------------------------

        prediction = (
            prediction >= THRESHOLD
        ).astype(np.uint8)

        # ----------------------------------
        # Prepare ground truth
        # ----------------------------------

        mask = (
            mask > 0
        ).astype(np.uint8)

        # ----------------------------------
        # Calculate metrics
        # ----------------------------------

        dice = calculate_dice(
            prediction,
            mask
        )

        iou = calculate_iou(
            prediction,
            mask
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

        print(
            f"{image_name:<18} "
            f"Dice: {dice:.4f}  "
            f"IoU: {iou:.4f}"
        )


# ==========================================
# FINAL RESULTS
# ==========================================

mean_dice = np.mean(
    dice_scores
)

mean_iou = np.mean(
    iou_scores
)

print("------------------------------------------")
print("")
print("FINAL TEST RESULTS")
print("==========================================")
print("Test images :", len(test_images))
print("Threshold   :", THRESHOLD)
print("Mean Dice   :", f"{mean_dice:.4f}")
print("Mean IoU    :", f"{mean_iou:.4f}")
print("==========================================")
print("")
print("Evaluation completed successfully.")
print("Model      : U-Net with ResNet34")
print("Loss       : Tversky Loss + BCE")
print("Encoder    : ImageNet pretrained")
print("Threshold  :", THRESHOLD)
print("==========================================")