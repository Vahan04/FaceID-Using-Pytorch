import torch

from faceid.models import FaceEmbeddingNet


def main():
    model = FaceEmbeddingNet()

    x = torch.rand(2, 3, 160, 160)

    y = model(x)

    print("=" * 40)
    print("Model Test")
    print("=" * 40)

    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {y.shape}")

    embedding_norm = torch.norm(y, p=2, dim=1)

    print(f"Embedding Norms: {embedding_norm}")

    print("=" * 40)


if __name__ == "__main__":
    main()