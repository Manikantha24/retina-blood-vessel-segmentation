import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import sys
import os


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.6

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "unet_resnet34_tversky_bce35.pth"
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load model
# -----------------------------

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


# -----------------------------
# Prediction function
# -----------------------------

def predict(image_path, output_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    image = image / 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    image = image.unsqueeze(0).to(device)


    # -----------------------------
    # Model prediction
    # -----------------------------

    with torch.no_grad():

        prediction = model(image)

        prediction = torch.sigmoid(prediction)

        prediction = (
            prediction > THRESHOLD
        ).float()


    # -----------------------------
    # Save prediction
    # -----------------------------

    prediction = (
        prediction.squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        output_path,
        prediction
    )

    print(
        f"Prediction saved to: {output_path}"
    )


# -----------------------------
# Command-line interface
# -----------------------------

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage: python src/predict.py "
            "<input_image> <output_image>"
        )

        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]

    predict(
        input_image,
        output_image
    )