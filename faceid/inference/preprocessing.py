from pathlib import Path

from PIL import Image


def preprocess_image(
    image_path: str | Path,
    transform,
):
    """
    Load and preprocess an image.
    """

    image = Image.open(image_path).convert("RGB")

    if transform is not None:
        image = transform(image)

    return image