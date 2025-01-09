# ResUNet++: TensorFlow Implementation for Study & Learning Purposes

Comparative Study Paper B/w Vision Transformer and ResUNet++: https://drive.google.com/drive/u/0/folders/1W9OXVYjOzbkRuh7Je-WkjZ7NQmvf6VGl

Original Repo link here: https://github.com/Pajamajoker/Polyp-Detection-ResUNetPlusPlus

This repository contains my implementation of the **ResUNet++** architecture for medical image segmentation, developed using **TensorFlow**. I created this repo for study and learning purposes, with minor custom modifications to better understand the architecture and experiment with its components.

The ResUNet++ model is based on **Deep Residual U-Net (ResUNet)**, which combines the benefits of deep residual learning and the classic U-Net architecture. In ResUNet++, I explore the use of **residual blocks**, **squeeze-and-excitation blocks**, **ASPP (Atrous Spatial Pyramid Pooling)**, and **attention blocks** to improve performance in medical image segmentation tasks.

## 1. Introduction

**ResUNet++** is an advanced architecture designed for **medical image segmentation**, which improves upon the original **ResUNet** by incorporating several novel components:

- **Residual Blocks**: Facilitate deep learning by using skip connections that help prevent the vanishing gradient problem.
- **Squeeze-and-Excitation Blocks**: Enhance the network’s capability by recalibrating channel-wise feature responses.
- **ASPP (Atrous Spatial Pyramid Pooling)**: Captures multi-scale contextual information using dilated convolutions.
- **Attention Blocks**: Focus on the most relevant features by applying attention mechanisms, improving segmentation accuracy.

For a more detailed description of the architecture, refer to the [ResUNet++ paper](https://arxiv.org/pdf/1911.07067.pdf).

## 2. Modifications for Study and Learning

To help with my learning and experimentation, I made the following modifications to the original implementation:

1. **Custom Dataset Support**: Adapted the data pipeline to work with custom medical image datasets for testing and training.
2. **Hyperparameter Adjustments**: Experimented with hyperparameters like learning rate, batch size, and optimizer settings to understand their impact on performance.
3. **Visualization Enhancements**: Added visualization tools to observe feature maps and the final segmentation outputs during training.
4. **Simplified Training Loop**: Modified the training process to include better logging and model checkpointing, making it easier to monitor the learning progress.
5. **Pretrained Weights for Fine-Tuning**: Incorporated the option to fine-tune the model on custom datasets using pretrained weights from the original ResUNet++ model.

## 3. Usage

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd ResUNet++
