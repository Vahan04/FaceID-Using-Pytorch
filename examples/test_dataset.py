from faceid.datasets.face_dataset import FaceDataset


def main():

    dataset = FaceDataset("data")

    print(f"Dataset size: {len(dataset)}")


if __name__ == "__main__":
    main()