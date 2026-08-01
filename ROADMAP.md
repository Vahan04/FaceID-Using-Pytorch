# FaceID Roadmap

This document tracks the development progress of FaceID and outlines future improvements.

---

# Version 1.0 (Current)

## Project Architecture

- [x] Modular project structure
- [x] Clean package organization
- [x] Configuration support
- [x] Examples directory

---

## Dataset

- [x] Custom FaceDataset
- [x] Automatic dataset scanning
- [x] Image preprocessing
- [x] Triplet sampling
- [x] Identity-based dataset split
- [x] Dataset statistics

---

## Model

- [x] CNN Embedding Network
- [x] Reusable ConvBlock
- [x] 128-dimensional embeddings
- [x] L2 normalization

---

## Loss Functions

- [x] Triplet Loss

---

## Training

- [x] Training pipeline
- [x] Validation pipeline
- [x] Learning rate scheduler
- [x] Model checkpointing
- [x] Best model saving
- [x] Training history

---

## Evaluation

- [x] Euclidean distance
- [x] Verification metrics
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1 Score

---

## Inference

- [x] Face verification
- [x] Embedding extraction

---

## Logging

- [x] TensorBoard integration
- [x] Training loss visualization
- [x] Validation loss visualization
- [x] Learning rate visualization

---

# Future Improvements

## Training

- [ ] Early Stopping
- [ ] Mixed Precision Training
- [ ] Gradient Accumulation

---

## Metric Learning

- [ ] Hard Triplet Mining
- [ ] Semi-Hard Triplet Mining
- [ ] Batch Hard Mining
- [ ] Contrastive Loss
- [ ] ArcFace Loss

---

## Models

- [ ] ResNet18 backbone
- [ ] ResNet34 backbone
- [ ] MobileNetV3 backbone
- [ ] Vision Transformer backbone

---

## Evaluation

- [ ] ROC Curve
- [ ] AUC
- [ ] FAR
- [ ] FRR
- [ ] Equal Error Rate (EER)

---

## Deployment

- [ ] ONNX export
- [ ] TorchScript export
- [ ] FastAPI REST API
- [ ] Docker support

---

## Engineering

- [ ] Unit tests
- [ ] Continuous Integration (GitHub Actions)
- [ ] Documentation website
- [ ] Automatic code formatting
- [ ] Benchmark scripts

---

# Long-Term Vision

FaceID aims to become a lightweight educational framework for learning face recognition, metric learning, and PyTorch engineering from scratch while maintaining clean software architecture and production-quality code.