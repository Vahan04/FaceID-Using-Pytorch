from torchvision import transforms

from faceid.datasets.face_dataset import FaceDataset


def main():

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    dataset = FaceDataset(
        root_dir="data/lfw_funneled",
        transform=transform,
    )

    print(f"Dataset size: {len(dataset)}")


if __name__ == "__main__":
    main()