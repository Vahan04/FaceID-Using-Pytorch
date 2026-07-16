from operator import index
from pathlib import Path
import random

from sqlalchemy import label
from streamlit import image
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    """
    Face Dataset for training with Triplet Loss.

    Expected structure:

    dataset/
        person_001/
            img1.jpg
            img2.jpg

        person_002/
            img1.jpg
            img2.jpg
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    def __init__(self, root_dir: str, transform=None):
        super().__init__()

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
                raise FileNotFoundError(
        f"Dataset directory '{self.root_dir}' does not exist."
    )
        self.transform = transform

        # Every image in the dataset
        self.image_paths = []

        # Integer label for every image
        self.labels = []

        # label -> list[Path]
        self.person_to_images = {}

        # person_name -> label
        self.person_to_label = {}

        # label -> person_name
        self.label_to_person = {}

        self.skipped_identities = 0

        self._scan_dataset()

        self._dataset_summary()
    
    def _scan_dataset(self):

        label = 0

        for person_dir in sorted(self.root_dir.iterdir()):

        # Skip anything that isn't a folder
            if not person_dir.is_dir():
                continue

        # Collect all image files for this identity
            images = [
            image_path
            for image_path in person_dir.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        # Skip identities with fewer than 2 images
            if len(images) < 2:
                self.skipped_identities += 1
                continue

            person_name = person_dir.name

            self.person_to_label[person_name] = label
            self.label_to_person[label] = person_name
            self.person_to_images[label] = images

            for image_path in images:
                self.image_paths.append(image_path)
                self.labels.append(label)

            label += 1
    
    
    def __len__(self):
        return len(self.image_paths)
    
    def _dataset_summary(self):

        image_counts = [
        len(images)
        for images in self.person_to_images.values()
    ]

        print("=" * 50)
        print("Dataset Summary")
        print("=" * 50)

        print(f"Images              : {len(self.image_paths)}")
        print(f"Identities          : {len(self.person_to_label)}")

        if len(self.person_to_label) > 0:
            avg_images = len(self.image_paths) / len(self.person_to_label)
        else:
            avg_images = 0

        print(f"Average/images ID   : {avg_images:.2f}")

        if image_counts:
            print(f"Min images          : {min(image_counts)}")
            print(f"Max images          : {max(image_counts)}")
        else:
            print("Min images          : 0")
            print("Max images          : 0")

        print(f"Skipped identities  : {self.skipped_identities}")

        print("=" * 50)
    def _load_image(self, image_path: Path) -> Image.Image:
        """
        Load an image from disk.
        """

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image
    
    def _sample_positive(self, label: int, anchor_path: Path) -> Path:
        """
        Randomly sample another image of the same identity.
        """

        images = self.person_to_images[label]

        positive = random.choice(images)

        while positive == anchor_path:
            positive = random.choice(images)

        return positive
    def _sample_negative(self, label: int) -> Path:

        """
        Randomly sample an image from a different identity.
        """

        labels = list(self.person_to_images.keys())

        negative_label = random.choice(labels)

        while negative_label == label:
            negative_label = random.choice(labels)

        while negative_label == label:
            negative_label = random.choice(list(self.person_to_images.keys()))

        negative = random.choice(self.person_to_images[negative_label])

        return negative
    
    def __getitem__(self, index):
        """
        Return one training triplet.
        """

        anchor_path = self.image_paths[index]

        label = self.labels[index]
        positive_path = self._sample_positive(label, anchor_path)
        negative_path = self._sample_negative(label)

        anchor_image = self._load_image(anchor_path)
        positive_image = self._load_image(positive_path)
        negative_image = self._load_image(negative_path)

        #return (anchor_image, positive_image, negative_image, label, anchor_path, positive_path, negative_path )
        return (anchor_image, positive_image, negative_image, label)
