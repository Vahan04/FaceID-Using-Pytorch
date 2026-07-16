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

print(anchor.shape)
print(positive.shape)
print(negative.shape)

print(label)