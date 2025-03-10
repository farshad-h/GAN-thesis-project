# -*- coding: utf-8 -*-

import logging
import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.models import inception_v3
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Clustering Metrics
def evaluate_clustering_metrics(embeddings, labels, n_clusters=10):
    """
    Evaluate embeddings using clustering metrics to measure the quality of clustering structures.

    Args:
        embeddings (torch.Tensor or np.ndarray): Input embeddings to cluster.
        labels (torch.Tensor or np.ndarray): Ground truth labels for the embeddings.
        n_clusters (int): Number of clusters for KMeans clustering.

    Returns:
        dict: Contains the following metrics:
            - ARI: Adjusted Rand Index, measures the similarity between clustering and ground truth.
            - Silhouette Score: Indicates how similar an object is to its own cluster compared to other clusters.
            - Calinski-Harabasz Index: Ratio of the sum of between-clusters dispersion and within-cluster dispersion.
            - Davies-Bouldin Index: Measures cluster separation, lower values are better.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    pred_labels = kmeans.labels_

    ari = adjusted_rand_score(labels, pred_labels)
    silhouette = silhouette_score(embeddings, pred_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, pred_labels)
    davies_bouldin = davies_bouldin_score(embeddings, pred_labels)

    logger.info(f"Adjusted Rand Index (ARI): {ari:.4f}")
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

    return {
        "ARI": ari,
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Davies-Bouldin Index": davies_bouldin,
    }

# k-NN Accuracy
def evaluate_embedding_quality(embeddings, labels, k=5):
    """
    Evaluate the quality of embeddings using k-NN accuracy.

    Args:
        embeddings (torch.Tensor or np.ndarray): Input embeddings to evaluate.
        labels (torch.Tensor or np.ndarray): Ground truth labels for the embeddings.
        k (int): Number of neighbors for k-NN classifier.

    Returns:
        float: k-NN accuracy score.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    logger.info(f"k-NN Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_embedding_quality_cv(embeddings, labels, k=5, cv=5):
    """
    Evaluate the quality of embeddings using k-NN accuracy with cross-validation.

    Args:
        embeddings (torch.Tensor or np.ndarray): Input embeddings to evaluate.
        labels (torch.Tensor or np.ndarray): Ground truth labels for the embeddings.
        k (int): Number of neighbors for k-NN classifier.
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: Mean and standard deviation of k-NN accuracy over cross-validation folds.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Initialize k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation
    cv_scores = cross_val_score(knn, embeddings, labels, cv=cv, scoring='accuracy')

    # Calculate mean and standard deviation of k-NN accuracy across folds
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    logger.info(f"k-NN Accuracy (CV) - Mean: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")
    return mean_accuracy, std_accuracy

# Visualization
def visualize_embeddings(embeddings, labels, method="tsne"):
    """
    Visualize embeddings using dimensionality reduction techniques (PCA or t-SNE).

    Args:
        embeddings (torch.Tensor or np.ndarray): High-dimensional embeddings to visualize.
        labels (torch.Tensor or np.ndarray): Ground truth labels used for color-coding the points.
        method (str): Dimensionality reduction method ('tsne' or 'pca').

    Visualization:
        - 2D scatter plot of the embeddings.
        - Points are color-coded based on their labels for interpretability.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2)
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter, label="Ground Truth Labels")
    plt.title(f"{method.upper()} Visualization of Embeddings")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.show()

# Overlay Clusters
def overlay_clusters(embeddings, labels, n_clusters=10, method="tsne"):
    """
    Visualize embeddings with overlaid cluster predictions from KMeans.

    Args:
        embeddings (torch.Tensor or np.ndarray): Input embeddings to visualize.
        labels (torch.Tensor or np.ndarray): Ground truth labels for the embeddings.
        n_clusters (int): Number of clusters for KMeans clustering.
        method (str): Dimensionality reduction method for visualization ('tsne' or 'pca').

    Visualization:
        - Embedding points are colored based on ground truth labels.
        - Cluster predictions are shown with overlayed colors.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2)
    embeddings_2d = reducer.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    clusters = kmeans.predict(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="viridis", alpha=0.5, label="Data"
    )
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap="jet", alpha=0.2, label="Clusters"
    )
    plt.legend()
    plt.colorbar()
    plt.show()

# Evaluate Multiple Embeddings
def evaluate_multiple_embeddings(embeddings, labels, n_clusters=10, k=5):
    """
    Evaluate multiple sets of embeddings using clustering metrics and k-NN accuracy.

    Args:
        embeddings (list): List of embeddings to evaluate.
        labels (torch.Tensor or np.ndarray): Ground truth labels for the embeddings.
        n_clusters (int): Number of clusters for clustering evaluation.
        k (int): Number of neighbors for k-NN classifier.

    Returns:
        dict: Results containing clustering metrics and k-NN accuracy for each embedding.
    """
    results = {}
    for i, embedding in enumerate(embeddings):
        embedding_shape = embedding.shape
        logger.info(f"\nEvaluating embedding {i+1}/{len(embeddings)} with shape: {embedding_shape}")

        metrics = evaluate_clustering_metrics(embedding, labels, n_clusters=n_clusters)
        knn_accuracy = evaluate_embedding_quality(embedding, labels, k=k)
        metrics["k-NN Accuracy"] = knn_accuracy

        results[f"Embedding_{i+1}"] = metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")

    return results

def evaluate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, f1

# Fréchet Inception Distance (FID)
def calculate_fid(real_embeddings, generated_embeddings):
    """
    Compute the Fréchet Inception Distance (FID).
    """
    mu1, sigma1 = np.mean(real_embeddings, axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = np.mean(generated_embeddings, axis=0), np.cov(generated_embeddings, rowvar=False)
    
    # Compute the squared difference of means
    diff = mu1 - mu2
    
    # Compute sqrt of product of covariance matrices using scipy for numerical stability
    covmean = sqrtm(sigma1 @ sigma2)
    
    # Check if the sqrt matrix has imaginary parts due to numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_inception_score(embeddings, num_splits=10):
    """
    Compute the Inception Score (IS).
    """
    p_yx = F.softmax(torch.tensor(embeddings), dim=1).numpy()
    p_y = np.mean(p_yx, axis=0)
    
    split_scores = []
    for i in range(num_splits):
        part = p_yx[i * (len(p_yx) // num_splits) : (i + 1) * (len(p_yx) // num_splits), :]
        kl = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        kl = np.mean(np.sum(kl, axis=1))
        split_scores.append(np.exp(kl))
    
    return np.mean(split_scores), np.std(split_scores)