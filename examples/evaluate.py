import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from faceid.datasets.face_dataset import FaceDataset
from faceid.models.embedding_net import FaceEmbeddingNet
from faceid.evaluation.distances import euclidean_distance
from faceid.evaluation.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)
from faceid.evaluation.roc import ROCEvaluator

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

dataset = FaceDataset(
    root_dir="data/lfw_funneled",
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

model = FaceEmbeddingNet().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
all_distances = []
all_labels = []

with torch.no_grad():

    for anchor, positive, negative, _ in loader:

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        positive_distance = euclidean_distance(
            anchor_embedding,
            positive_embedding,
        )

        all_distances.append(
            positive_distance.cpu()
        )

        all_labels.append(
            torch.ones(len(anchor))
        )
        negative_distance = euclidean_distance(
            anchor_embedding,
            negative_embedding,
        )

        all_distances.append(
            negative_distance.cpu()
        )

        all_labels.append(
            torch.zeros(len(anchor))
        )
distances = torch.cat(all_distances)

labels = torch.cat(all_labels)
threshold = 0.8
predictions = (
    distances <= threshold
).long()
acc = accuracy(
    distances,
    labels,
    threshold,
)

prec = precision(
    predictions,
    labels,
)

rec = recall(
    predictions,
    labels,
)

f1 = f1_score(
    predictions,
    labels,
)
roc = ROCEvaluator()

roc.compute(
    distances,
    labels,
)

auc = roc.auc()

roc.plot()
print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("=" * 50)