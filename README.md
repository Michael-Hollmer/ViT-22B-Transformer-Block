# ViT-22B-Transformer-Block
A PyTorch implementation of a Vision Transformer (ViT) architecture inspired by Google's ViT-22B. This project focuses on the optimized transformer block structure introduced in the paper, adapted for smaller models that can run on standard hardware. Includes an example training script for MNIST and modular components for easy experimentation.

# Vision Transformer (ViT-22B) Implementation in PyTorch

This repository contains a **PyTorch implementation of a Vision Transformer (ViT) block architecture inspired by Google's ViT-22B model**, adapted for smaller-scale training on standard hardware and datasets like MNIST.

---

## 🚀 Features

- **ViT-22B Block**:
  - Multi-head self-attention with query/key/value projections.
  - Layer normalization without biases (as per the paper).
  - Parallel MLP branch with GELU activation.
  - Residual connections for stability.

- **Full Vision Transformer Model**:
  - Patch embedding with learnable positional encodings.
  - Learnable class token for classification.
  - Configurable number of transformer blocks and attention heads.

- **Training Script**:
  - Example training loop for MNIST dataset.
  - Includes preprocessing (resize, grayscale to RGB, normalization).
  - Reports training loss and test accuracy.


## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- einops
- numpy
- matplotlib
- Pillow

## 🖼️ Model Overview

The model splits an image into patches, projects them into embeddings, and processes them through multiple transformer blocks. A learnable class token is used for classification.
Key Parameters:

- block_count: Number of transformer blocks (default: 8)
- patch_size: Size of each image patch (default: 16)
- embedding_dim: Dimension of patch embeddings (default: 128)
- attention_heads: Number of attention heads per block (default: 4)
- number_output_classes: Number of output classes (default: 10 for MNIST)

▶️ How to Run
Train the Vision Transformer on MNIST:

    python train.py

The script will:
Download MNIST dataset.
Resize images to 224×224 and convert to 3 channels.
Train for 5 epochs and report accuracy.