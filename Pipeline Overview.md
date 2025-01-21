# Pipeline Overview

This document provides a detailed explanation of the architecture, training strategies, and evaluation metrics used in the **Embeddings and GANs** project. The pipeline is divided into two main plans: **Plan 1** (Embedding-Based GANs) and **Plan 2** (Embedding Generation with GANs), with additional focus on embedding generation, evaluation, and the integration of advanced techniques like contrastive learning and normalizing flows.

## 1. Embedding Generation

Embeddings are the core of this project and are generated using various methods for both Plan 1 and Plan 2. These embeddings are used as input for GANs or as outputs to generate new embeddings.

- **Encoder Models**:
  - **Autoencoders (AE)**: Basic, Intermediate, and Advanced versions of autoencoders are implemented in `encoder_models.py`. These models learn a compressed representation of the input data.
  - **Variational Autoencoders (VAE)**: VAEs are employed to generate probabilistic embeddings, allowing for sampling from a learned distribution.
  
- **Contrastive Learning (CL)**:
  - **Loss Functions**: The NT-Xent Loss and Triplet Loss are implemented in `CL_loss_function.py` to learn robust embeddings. These loss functions help in refining the quality of the generated embeddings through contrastive learning.

- **Normalizing Flows**:
  - Embedding refinement is achieved using **Normalizing Flows (NF)**, which apply probabilistic transformations to improve the quality and structure of embeddings, making them more suitable for use in generative models.

- **Output**: The output from these methods is a set of high-dimensional embeddings, which are stored and used as inputs for GANs in subsequent steps.

---

## 2. Plan 1: Embedding-Based GANs

**Objective**: In Plan 1, the focus is on **data reconstruction** using embeddings as input. The goal is to train GANs that can regenerate the original data based on these embeddings.

- **GAN Models**:
  - **Simple GAN**: A traditional GAN architecture, where random noise and embeddings are fed to the generator to produce synthetic data.
  - **ACGAN**: An extension of GANs that adds conditional labels to both the generator and the discriminator. Embeddings serve as conditions.
  - **InfoGAN**: An information-theoretic extension of GANs that encourages learning interpretable features from the embeddings.

- **Training**:
  - The training process for Plan 1 is implemented in `plan1_gan_training.py`, which involves feeding the embeddings (and random noise) into the generator to create data samples that resemble the original dataset.
  - The Discriminator’s task is to differentiate between real and generated data, guiding the Generator’s learning.

- **Evaluation**:
  - The quality of the generated samples is evaluated based on metrics like **Inception Score (IS)** and **Frechet Inception Distance (FID)**.
  - Additional clustering metrics (ARI, Silhouette Score) can be used to assess the quality of the embeddings used in the GAN.

---

## 3. Plan 2: Embedding Generation with GANs

**Objective**: Plan 2 focuses on **embedding generation**, where GANs are used to learn a distribution over the embeddings, which can later be used to generate new samples of the original dataset.

- **GAN Models**:
  - **WGAN-GP (Wasserstein GAN with Gradient Penalty)**: This GAN variant improves training stability and is used in Plan 2 to generate high-quality embeddings.
  - **VAE-GAN**: Combines the benefits of VAEs and GANs, where the generator learns a distribution over embeddings, and the discriminator evaluates the quality of generated embeddings.
  - **Contrastive-GAN**: A specialized GAN model that uses contrastive loss functions to refine the generated embeddings.
  - **Dual-GAN** and **CycleGAN**: These architectures are employed for unpaired data generation, where embeddings are mapped to new domains while maintaining consistency.

- **Training**:
  - The training for Plan 2 is implemented in `plan2_gan_training.py`, where the GAN models focus on generating embeddings that can be used to reconstruct original data or generate new synthetic samples.
  - The generator and discriminator are trained jointly, with embeddings used to condition the generation process, resulting in refined latent spaces.

- **Evaluation**:
  - **Clustering Metrics**: To assess the quality of generated embeddings, metrics like **Adjusted Rand Index (ARI)**, **Silhouette Score**, **Calinski-Harabasz Index**, and **Davies-Bouldin Index** are used.
  - **k-NN Classification**: The generated embeddings can also be evaluated for classification tasks using k-Nearest Neighbors (k-NN) to assess the usefulness of the embeddings for downstream tasks.
  - **Visualization**: Embeddings are visualized using **t-SNE** and **PCA** to understand their structure and distribution in lower dimensions.

---

## 4. Embedding Evaluation and Refinement

Once embeddings are generated, it is essential to evaluate their quality and refine them for use in GANs and other tasks.

- **Clustering**:
  - Embeddings are evaluated using clustering techniques to assess their separability and structure. Popular clustering metrics such as **Silhouette Score** and **ARI** are used.
  
- **Classification**:
  - The quality of embeddings is further tested by training classifiers (e.g., k-NN) on the embeddings to check for classification accuracy.

- **Visualizations**:
  - Embedding quality is visually inspected using **PCA** and **t-SNE** plots, which allow for visual inspection of the separability and structure of the learned embeddings.

---

## 5. Results and Evaluation Metrics

The final evaluation of the models focuses on comparing generated samples, assessing the quality of embeddings, and ensuring that the GANs are learning meaningful representations.

- **Clustering and Visualizations**:
  - **PCA**, **t-SNE**, and **Silhouette Scores** are used to assess the quality of embeddings.
  
- **Generative Model Evaluation**:
  - Metrics like **Inception Score (IS)** and **Frechet Inception Distance (FID)** are used to evaluate the quality of the generated samples.
  
- **Embedding Refinement**:
  - The Normalizing Flows are used to improve the embeddings, ensuring they are more suitable for the GAN’s generative process.

---

## 6. Final Thoughts and Future Work

- **Integration of Additional Datasets**: Future work includes experimenting with datasets beyond MNIST, enabling the models to scale to more complex data types.
- **Exploring Hybrid Models**: Combining contrastive learning with generative models to improve both the quality and interpretability of the generated embeddings.
- **Advanced GAN Architectures**: Investigating advanced architectures like **StyleGAN** and **BigGAN** for improved sample quality and diversity.
