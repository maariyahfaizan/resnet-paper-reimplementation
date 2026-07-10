# Reimplementation of Deep Residual Learning for Image Recognition (He et al., 2015)

**Paper:** *Deep Residual Learning for Image Recognition*  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
CVPR 2016

---

# 1. Introduction

The objective of this project was to faithfully reproduce the CIFAR-10 experiments presented in *Deep Residual Learning for Image Recognition*, Section 4.2 "CIFAR-10 and Analysis", by He et al. 
Rather than using TensorFlow's built-in ResNet implementation, every component of the network was implemented manually using TensorFlow/Keras.

The primary goals of this project were:

- Understand why residual learning solves the degradation problem.
- Implement both the Plain CNN and Residual Network described in the paper.
- Reproduce the CIFAR-10 preprocessing pipeline.
- Reproduce the training schedule used by the authors.
- Compare experimental results with those reported in the original paper.

---
# Repository Structure

The project is organized into separate modules for model definitions, data preprocessing, training, evaluation, and experiment results.

```text
ResNet_Reimplementation/
│
├── additional_images/
│   └── Figures and diagrams used in the documentation
│
├── evaluation/
│   ├── evaluation_plain56.py      # Evaluates the trained Plain-56 model
│   └── evaluation_resnet.py       # Evaluates the trained ResNet-56 model
│
├── models/
│   ├── plain_cnn.py               # Plain CNN architecture
│   ├── resnet56.py                # ResNet-56 architecture
│   ├── plain56.keras              # Trained Plain-56 model
│   └── resnet56.keras             # Trained ResNet-56 model
│
├── results/
│   ├── plain56/
│   │   ├── accuracy.png
│   │   ├── loss.png
│   │   ├── history.json
│   │   └── test_results.txt
│   │
│   └── resnet56/
│       ├── accuracy.png
│       ├── loss.png
│       ├── history.json
│       └── test_results.txt
│
├── utils/
│   └── dataset.py                 # CIFAR-10 loading and preprocessing
│
├── implementation_report.md       # Project implementation report
├── requirements.txt               # Python dependencies
├── train.py                       # Training script
└── visualization.py               # Exploartion of Dataset
```

### Directory Overview

| Directory/File | Purpose |
|----------------|---------|
| `models/` | Contains the manually implemented Plain CNN and ResNet architectures along with the saved trained models. |
| `utils/` | Dataset loading, preprocessing, augmentation, and train/validation/test split. |
| `evaluation/` | Scripts used to evaluate the trained models on the CIFAR-10 test set. |
| `results/` | Stores training history, accuracy/loss plots, and final test results for each experiment. |
| `additional_images/` | Figures and images referenced by the README and implementation report. |
| `train.py` | Trains a selected model using the training schedule described in the ResNet paper. |
| `visualization.py` | Utility functions for plotting or visualizing results. |
| `implementation_report.md` | Detailed description of the implementation process, assumptions, experiments, and findings. |
| `requirements.txt` | Python packages required to reproduce the project. |

---
# 2. Background

Increasing the depth of neural networks does not always improve performance.

The ResNet paper demonstrated that deeper *plain* convolutional networks eventually become harder to optimize, producing higher training and testing errors despite having greater representational capacity. This phenomenon is known as the **degradation problem**.

Residual learning addresses this by allowing each block to learn a residual function

$$
F(x)=H(x)-x
$$

instead of directly learning

$$
H(x).
$$

The block output becomes

$$
H(x)=F(x)+x
$$

through an identity shortcut connection.

---

# 3. Paper Architecture

The CIFAR-10 architecture used in the paper consists of

```text
Input (32×32×3)
      │
      ▼
3×3 Conv (16)
      │
      ▼
Stage 1
9 Residual Blocks
16 Filters
      │
      ▼
Stage 2
First Block Stride = 2
32 Filters
      │
      ▼
Stage 3
First Block Stride = 2
64 Filters
      │
      ▼
Global Average Pooling
      │
      ▼
Fully Connected (10)
      │
      ▼
Softmax
```

Each residual block contains

```text
Input
  │
  ▼
Conv 3×3
  │
  ▼
BatchNorm
  │
  ▼
ReLU
  │
  ▼
Conv 3×3
  │
  ▼
BatchNorm
  │
  ▼
Add Shortcut
  │
  ▼
ReLU
```

The network depth is defined by

$$
6n+2
$$

which gives

| n | Depth |
|---:|------:|
| 3 | 20 |
| 5 | 32 |
| 7 | 44 |
| 9 | 56 |
| 18 | 110 |

This implementation reproduces **ResNet-56**, i.e, n = 9
---

# 4. Dataset

The experiments use **CIFAR-10**, exactly as described in Section 4.2 of the paper.

| Split | Images |
|-------|-------:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

---

# 5. Data Preprocessing

The preprocessing pipeline follows the paper.

### Training

- Convert images to `float32`
- Compute the mean image from the training set
- Subtract the mean image
- Pad every image by 4 pixels
- Random crop back to **32×32**
- Random horizontal flip

### Testing

- Convert images to `float32`
- Subtract the training mean image

TensorFlow's `tf.data.Dataset` API was used with

- `shuffle()`
- `batch(128)`
- `prefetch()`

---

# 6. Plain Network Implementation

The Plain CNN uses the same architecture as ResNet but removes the shortcut connections.

Each block consists of

```text
Conv
 │
 ▼
BatchNorm
 │
 ▼
ReLU
 │
 ▼
Conv
 │
 ▼
BatchNorm
 │
 ▼
ReLU
```

This model serves as the baseline for demonstrating the degradation problem.

---

# 7. Residual Network Implementation

Residual blocks were implemented manually using the Keras Functional API.

Each block computes

```text
F(x)
 │
 ▼
Add Shortcut
 │
 ▼
ReLU
```

rather than directly learning the desired mapping.

---

# 8. Option A Shortcut

One of the goals of this project was to remain faithful to the paper.

Most modern ResNet implementations use **projection shortcuts (Option B)** involving **1×1 convolutions**.

However, the paper explicitly states that **Option A** is used for CIFAR-10.

Therefore the shortcut implementation performs

1. Downsample by taking every second pixel

```python
shortcut[:, ::2, ::2, :]
```

2. Zero-pad the channel dimension using

```python
tf.pad(...)
```

Conceptually, the shortcut performs

```text
32×32×16
      │
      ▼
Take every second pixel
(::2, ::2)
      │
      ▼
16×16×16
      │
      ▼
Zero-pad channels
      │
      ▼
16×16×32
```

These TensorFlow operations were wrapped inside Keras `Lambda` layers because the Functional API requires graph-compatible operations. However, the Lambda Layers later presented difficulty in loading the model and instead the saved weights from the model were used for evaluation. 

---

# 9. Training Configuration

The paper specifies

| Hyperparameter | Value |
|---------------|------:|
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| Batch Size | 128 |
| Dropout | None |

Initial learning rate

```text
0.1
```

Learning rate schedule

```text
0.1
 ↓
0.01
 ↓
0.001
```

---

# 10. Converting Iterations to Epochs

Unlike many modern papers, ResNet reports learning rate changes in **iterations** rather than epochs.

The paper specifies learning-rate changes at

- 32,000 iterations
- 48,000 iterations
- 64,000 iterations

The CIFAR-10 training set contains

```text
45,000 images
```

Using a batch size of

```text
128
```

gives

$$
\frac{45000}{128}=351.56
$$

or approximately **352 iterations per epoch**.

Therefore,

$$
\frac{32000}{352}\approx91
$$

$$
\frac{48000}{352}\approx136
$$

$$
\frac{64000}{352}\approx182
$$

This implementation therefore used

| Epochs | Learning Rate |
|---------|--------------:|
| 0–90 | 0.1 |
| 91–135 | 0.01 |
| 136–182 | 0.001 |

for a total of **182 epochs**.

---

# 11. Implementation Decisions

Some implementation details were not explicitly specified in the paper and required design decisions during reproduction.

### Weight Initialization

Following the paper, He initialization was used for all convolutional layers via TensorFlow's `he_normal` initializer.

### Padding

The paper does not explicitly specify the padding strategy. TensorFlow's `padding="same"` was used to preserve the feature-map sizes described in the architecture.

### Batch Normalization

The paper specifies the use of Batch Normalization but does not define its hyperparameters. TensorFlow's default `BatchNormalization` configuration was therefore used.

### Validation Split

Following the paper, the CIFAR-10 training set was divided into **45,000 training images** and **5,000 validation images**.

### Option A Implementation

The paper specifies the use of Option A shortcuts but does not describe a framework-specific implementation. In TensorFlow/Keras, Option A was implemented using `Lambda` layers to perform spatial downsampling and channel zero-padding.

---

# 12. Challenges and Mistakes

This project involved several implementation and debugging challenges.

## Understanding Option A

Initially, the shortcut implementation was unclear because the paper only briefly describes Option A.

Additional study was required to understand how spatial downsampling and zero-padding should be performed.

---
## Incorrect Learning Rate Schedule

An early implementation used **164 epochs** before converting the paper's iteration schedule into epochs.

This was corrected to **182 epochs**.

---

## Training the Wrong Model

One complete 182-epoch training run accidentally imported

```python
from models.plain_cnn import model
```

instead of

```python
from models.resnet56 import model
```

This resulted in training the Plain-56 architecture twice before the mistake was identified.

---

## Keras Lambda Serialization

The use of `Lambda` layers caused issues when loading the saved model because Keras 3 restricts deserialization of anonymous Python functions by default.

The issue was resolved during evaluation by reconstructing the architecture and loading the trained weights.

---

## CPU-only Training

TensorFlow GPU acceleration was unavailable because TensorFlow 2.21 no longer supports CUDA on native Windows.

Training was therefore performed entirely on CPU, resulting in training times of approximately **35 hours per model**.

---

# 13. Results

| Model | Paper Test Error | Reproduced Test Error |
|--------|-----------------:|----------------------:|
| Plain-56 | 13.67% | **13.14%** |
| ResNet-56 | 6.97% | **7.44%** |

The reproduced results closely match those reported in the original paper, with the reproduced ResNet-56 differing by only **0.47 percentage points** from the published result.

## Plain-56 Training Accuracy

The following figure shows the training and validation accuracy throughout training. <br> 

**Note:** The titles of the Plain-56 accuracy and loss figures are incorrectly labeled as "ResNet-56". This is a labeling error only, the data shown in the figures correspond to the Plain-56 model.

![Plain-56 Accuracy](results/plain56/accuracy.png)

---

## Plain-56 Training Loss

The following figure shows the training and validation loss throughout training.

![Plain-56 Loss](results/plain56/loss.png)

---

## ResNet-56 Training Accuracy

The following figure shows the training and validation accuracy of the residual network.

![ResNet-56 Accuracy](results/resnet56/accuracy.png)

---

## ResNet-56 Training Loss

The following figure shows the training and validation loss of the residual network.

![ResNet-56 Loss](results/resnet56/loss.png)

---

# 14. Lessons Learned

This project provided practical experience in

- Reading and interpreting a deep learning research paper.
- Translating mathematical descriptions into TensorFlow implementations.
- Reproducing published experimental protocols.
- Debugging architecture and training issues.
- Understanding residual learning and identity mappings.
- Appreciating the importance of reproducibility in machine learning research.

---

# References

K. He, X. Zhang, S. Ren, and J. Sun, **Deep Residual Learning for Image Recognition**, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.