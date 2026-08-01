<div align="center">

# 🚀 FaceID

### A Modular Face Recognition Framework Built with PyTorch

Train deep face embeddings using **Triplet Loss**, evaluate face similarity, perform inference, and experiment with metric learning through a clean, modular, and extensible architecture.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-v1.0-success.svg)

</p>

</div>

---

# 📖 Overview

**FaceID** is an end-to-end face recognition framework implemented from scratch using **PyTorch**.

Unlike traditional image classification models, FaceID learns a **feature embedding space** using **Metric Learning**. Images of the same person are mapped close together, while images of different people are pushed farther apart using **Triplet Loss**.

The project was built to understand every component behind modern face recognition systems instead of relying on high-level libraries.

It focuses on both:

- Deep Learning
- Software Engineering

making it suitable as both an educational project and a strong portfolio repository.

---

# ✨ Features

- ✅ Modular project architecture
- ✅ Custom FaceDataset
- ✅ Automatic dataset scanning
- ✅ Identity-based dataset splitting
- ✅ CNN Face Embedding Network
- ✅ L2 Normalized Embeddings
- ✅ Triplet Loss
- ✅ Training Pipeline
- ✅ Validation Pipeline
- ✅ TensorBoard Logging
- ✅ Checkpoint Saving
- ✅ Learning Rate Scheduler
- ✅ Evaluation Metrics
- ✅ Face Verification
- ✅ Inference Module

---

# 🏗 Architecture

```
                    Dataset
                       │
                       ▼
               Triplet Sampler
                       │
                       ▼
            Face Embedding Network
                       │
                       ▼
              128-D Face Embeddings
                       │
                       ▼
                 Triplet Loss
                       │
                       ▼
                  Optimizer
                       │
                       ▼
             Learned Embedding Space
```

---

# 📂 Project Structure

```text
FaceID/

├── faceid/
│
├── datasets/
│   ├── face_dataset.py
│   ├── split.py
│   └── __init__.py
│
├── models/
│   ├── blocks.py
│   └── embedding_net.py
│
├── losses/
│   └── triplet_loss.py
│
├── trainers/
│   └── trainer.py
│
├── evaluation/
│   ├── metrics.py
│   ├── distances.py
│   └── verifier.py
│
├── inference/
│   └── recognizer.py
│
├── utils/
│
├── examples/
│   ├── train.py
│   └── evaluate.py
│
├── configs/
├── tests/
├── checkpoints/
├── README.md
└── ROADMAP.md
```

---

# 📊 Training Pipeline

```
Dataset
   │
DataLoader
   │
Triplet Sampling
   │
Forward Pass
   │
Triplet Loss
   │
Backward Pass
   │
Optimizer
   │
Validation
   │
Checkpoint
```

---

# 🧠 Model

The Face Embedding Network is composed of multiple convolutional blocks followed by a projection layer.

```
Input Image
      │
Conv Block
      │
Conv Block
      │
Conv Block
      │
Conv Block
      │
Adaptive Average Pooling
      │
Fully Connected Layer
      │
128-D Embedding
      │
L2 Normalization
```

The final output is a **128-dimensional embedding vector** suitable for face verification.

---

# 📁 Dataset Format

Expected dataset structure:

```text
dataset/

person_001/
    img1.jpg
    img2.jpg
    img3.jpg

person_002/
    img1.jpg
    img2.jpg
```

Each directory represents one identity.

During training, the dataset automatically generates:

- Anchor
- Positive
- Negative

triplets for Triplet Loss optimization.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FaceID.git
```

Go to the project

```bash
cd FaceID
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋 Training

Run training

```bash
python -m examples.train
```

During training the framework automatically:

- trains the model
- validates every epoch
- logs TensorBoard metrics
- saves the best checkpoint
- tracks training history

---

# 📈 TensorBoard

Launch TensorBoard

```bash
python -m tensorboard.main --logdir=runs
```

Open

```
http://localhost:6006
```

TensorBoard includes:

- Training Loss
- Validation Loss
- Learning Rate

---

# 🖼 Training Dashboard

<p align="center">

<img src="assets/faceid_dashboard.png" width="100%">

</p>

---

# 📏 Evaluation

Run evaluation

```bash
python -m examples.evaluate
```

Implemented metrics

- Euclidean Distance
- Verification Accuracy
- Precision
- Recall
- F1 Score

---

# 🔍 Inference

The inference pipeline compares two images.

```
Image A
     │

Image B
     │

Embedding Extraction
     │

Euclidean Distance
     │

Same Person?
```

---

# 📦 Checkpoints

The framework automatically saves

```
checkpoints/

best_model.pth
```

including

- Model weights
- Optimizer state
- Scheduler state
- Training loss
- Validation loss

---

# 📚 Technologies

- Python
- PyTorch
- Torchvision
- Pillow
- TensorBoard

---

# 📈 Current Results

| Component | Status |
|-----------|--------|
| Dataset | ✅ |
| Identity Split | ✅ |
| CNN Backbone | ✅ |
| Triplet Loss | ✅ |
| Training | ✅ |
| Validation | ✅ |
| TensorBoard | ✅ |
| Checkpoint Saving | ✅ |
| Evaluation | ✅ |
| Inference | ✅ |

---

# 🎯 Learning Objectives

This project was built to deepen understanding of:

- Metric Learning
- Face Recognition
- PyTorch
- Triplet Loss
- Software Architecture
- Deep Learning Training Pipelines
- Modular Machine Learning Projects

---

# 🔮 Future Improvements

Possible future work:

- Hard Triplet Mining
- Semi-Hard Triplet Mining
- ResNet Backbone
- MobileNet Backbone
- Vision Transformer
- ONNX Export
- TorchScript Export
- FastAPI Deployment
- Docker Support
- Mixed Precision Training

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve the project, feel free to open an Issue or submit a Pull Request.

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star!

### Built with ❤️ using PyTorch

**Author:** Vahan Arubjanyan

GitHub: https://github.com/Vahan04

</div>