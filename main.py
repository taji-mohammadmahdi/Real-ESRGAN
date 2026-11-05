from pathlib import Path

import cv2
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input" / "photo.jpg"
OUTPUT_DIR = BASE_DIR / "enhanced"
WEIGHTS_DIR = BASE_DIR / "weights"

OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)


def main():

    # Select processing device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Model configuration
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    model_url = (
        "https://github.com/xinntao/Real-ESRGAN/"
        "releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )

    # Create upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_url,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    # Check input image
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_PATH}"
        )

    # Read image
    image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read input image.")

    print("Enhancing image...")

    # Image enhancement
    output, _ = upsampler.enhance(
        image,
        outscale=4
    )

    # Save result
    output_path = OUTPUT_DIR / "photo_4x.png"

    success = cv2.imwrite(
        str(output_path),
        output
    )

    if not success:
        raise RuntimeError("Could not save output image.")

    print(f"Done! Saved to: {output_path}")


if __name__ == "__main__":
    main()
