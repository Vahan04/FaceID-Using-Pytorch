from torchvision import transforms

from faceid.datasets.face_dataset import FaceDataset


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


dataset = FaceDataset(
    "data/lfw_funneled",
    transform=transform,
)

anchor, positive, negative, label = dataset[0]

print(f"Anchor   : {anchor.shape}")
print(f"Positive : {positive.shape}")
print(f"Negative : {negative.shape}")
print(f"Label    : {label}")
print(anchor.min(), anchor.max())