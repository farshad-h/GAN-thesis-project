# Embeddings and GANs: Advanced Architectures and Applications

This repository explores advanced GAN architectures and their integration with embeddings for tasks such as image reconstruction, cross-domain learning, contrastive learning, and generative modeling using Normalizing Flows (NF) and self-supervised learning methods like SimCLR.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Pipeline Overview](#pipeline-overview)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project investigates the integration of embeddings with GANs using advanced architectures and training strategies. Key focus areas include:
1. **Plan 1**: GANs that reconstruct original data using embeddings as input.
2. **Plan 2**: GANs that generate embeddings, later used for generating new samples of the original dataset.
3. **Contrastive Learning**: Learning robust embeddings using methods like NT-Xent and Triplet Loss.
4. **Normalizing Flows**: Refining embeddings using probabilistic transformations.
5. **Self-Supervised Learning**: Employing SimCLR for representation learning.

## Repository Structure

```
root/
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── pipeline_overview.md        # Detailed pipeline and architecture explanation
├── data/                       # Datasets directory
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed datasets
│   └── README.md               # Data directory documentation
├── src/                        # Source code for core tasks
│   ├── __init__.py             # Marks src as a package
│   ├── data_utils.py           # Data loading and preprocessing utilities
│   ├── evaluation.py           # Evaluation utilities
│   ├── embeddings/             # Embedding generation components
│   │   ├── embedding_generation.py
│   │   ├── encoder_training.py
│   │   └── CL_loss_function.py
│   ├── gan_workflows/          # GAN-related workflows
│   │   ├── plan1/              # Plan 1 (Reconstruction)
│   │   │   ├── plan1_gan_models.py
│   │   │   ├── plan1_gan_training.py
│   │   │   └── main_plan1_gan_training.ipynb
│   │   ├── plan2/              # Plan 2 (Embedding Generation)
│   │   │   ├── plan2_gan_models.py
│   │   │   ├── plan2_gan_training.py
│   │   │   └── main_plan2_gan_training.ipynb
│   └── utils/                  # Shared utilities
│       ├── logger.py           # Logging utilities
│       └── __init__.py
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── embedding_generation.ipynb
│   ├── plan1_experiments.ipynb
│   ├── plan2_experiments.ipynb
│   └── visualization.ipynb     # Visualizations and analysis
├── tests/                      # Unit and integration tests
│   ├── test_data_utils.py
│   ├── test_evaluation.py
│   ├── test_embeddings.py
│   ├── test_gan_workflows.py
│   └── test_contrastive_loss.py
├── results/                    # Results and saved models
│   ├── embeddings/             # Saved embeddings
│   ├── models/                 # Saved GAN models
│   └── visualizations/         # Generated plots and images
└── logs/                       # Logs for training and evaluation
    ├── plan1/
    ├── plan2/
    └── embedding_generation/
```

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib
- OpenCV (for SIFT feature extraction)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/farshad-h/GAN-thesis-project.git
   cd GAN-thesis-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place your dataset in the `data/raw/` directory.
2. Use `data_utils.py` to preprocess and save processed datasets in `data/processed/`.

## Pipeline Overview

See `pipeline_overview.md` for a detailed explanation of the architecture, training strategies, and metrics.

## Features

- **GANs with Embeddings**:
  - Plan 1 and Plan 2 GANs for data reconstruction and embedding generation.
- **Embedding Refinement**:
  - Use Normalizing Flows to refine embeddings probabilistically.
- **Self-Supervised Learning**:
  - SimCLR for learning robust representations.
- **Contrastive Learning**:
  - NT-Xent and Triplet Loss for embedding enhancement.
- **Evaluation Metrics**:
  - Clustering metrics (e.g., Silhouette Score, ARI).
  - Classification accuracy and visualizations.

## Usage

### Generating Embeddings

Run the embedding generation notebook:
```bash
jupyter notebook notebooks/embedding_generation.ipynb
```

### Training GANs

1. Navigate to the desired plan directory (`plan1/` or `plan2/`).
2. Modify hyperparameters in `.py` or `.ipynb` files as needed.
3. Train the GANs:
   ```bash
   python src/gan_workflows/plan1/plan1_gan_training.py
   ```

### Contrastive Learning

1. Use `CL_loss_function.py` for contrastive loss computations.
2. Integrate with encoders or embeddings:
   ```python
   from src.embeddings.CL_loss_function import NTXentLoss, TripletLoss
   ```

### Evaluating Models

Use `evaluation.py` to assess the quality of embeddings and GAN outputs:
```bash
python src/evaluation.py
```

## Results

- **Embeddings**:
  - Visualizations and clustering results for PCA, SVD, and NMF.
  - Self-supervised embeddings using SimCLR.
- **GANs**:
  - Reconstruction and generation quality metrics.
  - Visual comparisons of generated samples.

## Future Work

- Integrate additional datasets beyond MNIST.
- Experiment with advanced GAN architectures like StyleGAN.
- Explore hybrid approaches combining contrastive learning with generative modeling.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for training and evaluation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

