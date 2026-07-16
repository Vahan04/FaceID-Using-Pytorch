import matplotlib.pyplot as plt
import torch
import random
from torchvision import transforms

from faceid.datasets.face_dataset import FaceDataset




def tensor_to_image(tensor):
    return tensor.permute(1, 2, 0).numpy()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])
dataset = FaceDataset(
    "data/lfw_funneled",
    transform=transform,
)
index = random.randint(0, len(dataset) - 1)
(anchor, positive, negative, label, anchor_path, positive_path, negative_path) = dataset[index]

anchor_image = tensor_to_image(anchor)
positive_image = tensor_to_image(positive)
negative_image = tensor_to_image(negative)
print(f"Label: {label}")

print()

print(f"Anchor   : {anchor_path}")
print(f"Positive : {positive_path}")
print(f"Negative : {negative_path}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(anchor_image)
axes[0].set_title("Anchor")
axes[0].axis("off")

axes[1].imshow(positive_image)
axes[1].set_title("Positive")
axes[1].axis("off")

axes[2].imshow(negative_image)
axes[2].set_title("Negative")
axes[2].axis("off")

plt.tight_layout()
plt.show()