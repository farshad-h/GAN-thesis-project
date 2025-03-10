# -*- coding: utf-8 -*-

import numpy as np
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist

# Set up the logging configuration to remove the timestamp and INFO:root prefix
logger = logging.getLogger()  # Use the default logger
logger.setLevel(logging.INFO)  # Set logging level to INFO

# Create a custom StreamHandler with the desired format
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')  # Removed timestamp and INFO:root
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.handlers = []  # Remove default handlers
logger.addHandler(stream_handler)

def preprocess_images(images):
    """
    Normalize the images to [-1, 1] range
    """
    num_images, width, height = images.shape
    images = images.reshape(num_images, 1, width, height).astype('float32')
    images = (images - 127.5) / 127.5
    return images

def load_mnist_data(fraction=1.0, batch_size=64, shuffle=True):
    """
    Load and preprocess the MNIST dataset.

    Args:
        fraction (float): Fraction of the dataset to use.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Combined train and test DataLoader.
    """
    # Load the MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    # Combine train and test datasets
    combined_x = np.concatenate([mnist_train.data.numpy(), mnist_test.data.numpy()])
    combined_y = np.concatenate([mnist_train.targets.numpy(), mnist_test.targets.numpy()])

    # Normalize images
    combined_x = preprocess_images(combined_x)

    # Use a fraction of the dataset for faster experimentation
    sample_indices = np.random.choice(len(combined_x), int(fraction * len(combined_x)), replace=False)
    sampled_x = combined_x[sample_indices]
    sampled_y = combined_y[sample_indices]

    print('Sampled Dataset:', sampled_x.shape, sampled_y.shape)

    # Create a DataLoader
    dataset = TensorDataset(torch.tensor(sampled_x), torch.tensor(sampled_y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def split_dataset(data, labels, validation_split=0.2):
    """
    Split the dataset into training and validation sets using stratified splitting.

    Args:
        data (numpy.ndarray): Input data.
        labels (numpy.ndarray): Corresponding labels.
        validation_split (float): Fraction of data to use for validation.

    Returns:
        tuple: Training and validation sets (data and labels).
    """
    data_flat = data.reshape(data.shape[0], -1)  # Flatten for splitting

    # Stratified split to prevent label leakage
    train_x, val_x, train_y, val_y = train_test_split(
        data_flat, labels, test_size=validation_split, stratify=labels, random_state=42
    )

    # Reshape data back to original dimensions
    train_x = train_x.reshape(-1, data.shape[1], data.shape[2])
    val_x = val_x.reshape(-1, data.shape[1], data.shape[2])

    print("Train Dataset:", train_x.shape, train_y.shape)
    print("Validation Dataset:", val_x.shape, val_y.shape)

    return (train_x, train_y), (val_x, val_y)

def load_data(data_arrays, batch_size, shuffle=True):
    """
    Constructs a PyTorch DataLoader with shuffling and batching.

    Args:
        data_arrays (list of np.ndarray): List of data arrays (e.g., [X, y]).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the provided data.
    """
    # Convert NumPy arrays to PyTorch tensors
    tensors = [torch.from_numpy(arr) for arr in data_arrays]
    dataset = TensorDataset(*tensors)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def load_embeddings(embedding_file, device, batch_size=64):
    """
    Loads embeddings and their associated labels from a specified file and 
    creates a DataLoader for batching the embeddings.

    Args:
        embedding_file (str): Path to the file containing embeddings and labels.
        device (torch.device): The device (CPU/GPU) to load the tensors onto.
        batch_size (int, optional): The batch size for DataLoader. Default is 64.

    Returns:
        tuple: A tuple containing:
            - embeddings (torch.Tensor): Loaded embeddings.
            - labels (torch.Tensor): Corresponding labels for the embeddings.
            - data_loader (DataLoader): DataLoader for batching embeddings.
    """
    logger.info(f"Loading embeddings from: {embedding_file}")
    data = torch.load(embedding_file)
    embeddings = data["embeddings"].to(device)
    labels = data["labels"].to(device)

    # Create DataLoader for embeddings
    data_loader = DataLoader(embeddings, batch_size=batch_size, shuffle=True)

    return embeddings, labels, data_loader

def save_embeddings(embeddings, labels, file_path):
    """
    Saves embeddings and labels to a file.

    Args:
        embeddings (torch.Tensor): Embeddings to save.
        labels (torch.Tensor): Corresponding labels.
        file_path (str): Path to save the file.
    """
    data = {"embeddings": embeddings, "labels": labels}
    torch.save(data, file_path)
    logger.info(f"Embeddings saved to: {file_path}")

def save_embeddings(embeddings, labels, file_path, save_format):
    """Save embeddings and labels in the specified format."""
    if save_format == "pt":
        torch.save({"embeddings": embeddings, "labels": labels}, file_path)
    elif save_format == "npy":
        np.save(file_path, {"embeddings": embeddings.numpy(), "labels": labels.numpy()})
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

def analyze_embeddings(embeddings, expected_dim=None, labels=None):
    """
    Analyzes the embeddings by providing detailed statistical information, 
    dimensional checks, and other useful diagnostics like sparsity, skewness, 
    kurtosis, pairwise distance, and outlier detection.

    Args:
        embeddings (torch.Tensor): The tensor of embeddings to analyze.
        expected_dim (int, optional): The expected dimensionality of the embeddings.
        labels (torch.Tensor, optional): The labels associated with the embeddings 
                                         (used for cosine similarity checks).

    Raises:
        ValueError: If embeddings are invalid (empty, NaN, inf values, or wrong dimensionality).
    """
    # Check if embeddings are None or empty
    if embeddings is None or embeddings.numel() == 0:
        raise ValueError("Embeddings are empty or not properly generated.")
    
    # Log basic information
    logger.info(f"Embeddings are of shape: {embeddings.shape}")
    logger.info(f"Data type: {embeddings.dtype}")
    logger.info(f"Device: {embeddings.device}")

    # Check for NaN and infinity values
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or infinite values.")

    # Check if embeddings are 2D
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D tensor.")

    # Optionally check the dimensionality (if expected_dim is provided)
    if expected_dim and embeddings.size(1) != expected_dim:
        raise ValueError(f"Embeddings should have {expected_dim} dimensions, but got {embeddings.size(1)}.")
    
    # Statistical summary of the embeddings
    logger.info(f"Mean: {torch.mean(embeddings)}")
    logger.info(f"Standard Deviation: {torch.std(embeddings)}")
    logger.info(f"Min: {torch.min(embeddings)}")
    logger.info(f"Max: {torch.max(embeddings)}")
    logger.info(f"Median: {torch.median(embeddings)}")

    # Additional insights:
    # 1. L2 Norm (Magnitude) of each embedding
    norms = torch.norm(embeddings, p=2, dim=1)
    logger.info(f"Mean L2 Norm: {torch.mean(norms)}")
    logger.info(f"Standard Deviation of L2 Norms: {torch.std(norms)}")

    # 2. Size Consistency: Check if embeddings are of the same size
    embedding_size = embeddings.size(1)
    logger.info(f"Each embedding has {embedding_size} dimensions.")

    # 3. Sparsity: Check the proportion of non-zero elements
    sparsity = torch.count_nonzero(embeddings) / embeddings.numel()
    logger.info(f"Sparsity (proportion of non-zero elements): {sparsity.item()}")

    # 4. Skewness & Kurtosis of the embeddings
    embedding_numpy = embeddings.cpu().detach().numpy()  # Convert to numpy for skew/kurtosis calculation
    skewness = skew(embedding_numpy.flatten())
    kurt = kurtosis(embedding_numpy.flatten())
    logger.info(f"Skewness of embeddings: {skewness}")
    logger.info(f"Kurtosis of embeddings: {kurt}")

    # 5. Pairwise Distance (Euclidean)
    pairwise_distances = pdist(embedding_numpy, 'euclidean')  # Or use 'cosine' distance
    logger.info(f"Pairwise distance (mean): {np.mean(pairwise_distances)}")

    # 6. Cosine Similarity with True Labels (if labels available)
    if labels is not None:
        # Reshape labels to match embeddings' dimension for cosine similarity
        labels_reshaped = labels.unsqueeze(1).expand_as(embeddings)  # Add a dimension to match
        cos_sim = torch.nn.functional.cosine_similarity(embeddings, labels_reshaped, dim=1)
        logger.info(f"Average Cosine similarity with true labels: {cos_sim.mean().item()}")

    # 7. Outliers Detection (Using Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    outlier_flags = lof.fit_predict(embedding_numpy)  # -1 for outliers, 1 for inliers
    num_outliers = np.sum(outlier_flags == -1)
    logger.info(f"Number of outliers detected in embeddings: {num_outliers}")

    logger.info("Embeddings analysis completed.")

def analyze_embeddings_v2(embeddings, expected_dim=None, labels=None):
    """
    Analyzes the embeddings by providing detailed statistical information, 
    dimensional checks, and other useful diagnostics like sparsity, skewness, 
    kurtosis, pairwise distance, and outlier detection.

    Args:
        embeddings (torch.Tensor): The tensor of embeddings to analyze.
        expected_dim (int, optional): The expected dimensionality of the embeddings.
        labels (torch.Tensor, optional): The labels associated with the embeddings 
                                         (used for cosine similarity checks).

    Raises:
        ValueError: If embeddings are invalid (empty, NaN, inf values, or wrong dimensionality).
    """
    # Check if embeddings are None or empty
    if embeddings is None or embeddings.numel() == 0:
        raise ValueError("Embeddings are empty or not properly generated.")
    
    # Log basic information
    logger.info(f"Embeddings are of shape: {embeddings.shape}")
    logger.info(f"Data type: {embeddings.dtype}")
    logger.info(f"Device: {embeddings.device}")

    # Check for NaN and infinity values
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or infinite values.")

    # Check if embeddings are 2D
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D tensor.")

    # Optionally check the dimensionality (if expected_dim is provided)
    if expected_dim and embeddings.size(1) != expected_dim:
        raise ValueError(f"Embeddings should have {expected_dim} dimensions, but got {embeddings.size(1)}.")
    
    # Statistical summary of the embeddings
    logger.info(f"Mean: {torch.mean(embeddings)}")
    logger.info(f"Standard Deviation: {torch.std(embeddings)}")
    logger.info(f"Min: {torch.min(embeddings)}")
    logger.info(f"Max: {torch.max(embeddings)}")
    logger.info(f"Median: {torch.median(embeddings)}")

    # Additional insights:
    # 1. L2 Norm (Magnitude) of each embedding
    norms = torch.norm(embeddings, p=2, dim=1)
    logger.info(f"Mean L2 Norm: {torch.mean(norms)}")
    logger.info(f"Standard Deviation of L2 Norms: {torch.std(norms)}")

    # 2. Size Consistency: Check if embeddings are of the same size
    embedding_size = embeddings.size(1)
    logger.info(f"Each embedding has {embedding_size} dimensions.")

    # 3. Sparsity: Check the proportion of non-zero elements
    sparsity = torch.count_nonzero(embeddings) / embeddings.numel()
    logger.info(f"Sparsity (proportion of non-zero elements): {sparsity.item()}")

    # 4. Skewness & Kurtosis of the embeddings
    embedding_numpy = embeddings.cpu().detach().numpy()  # Convert to numpy for skew/kurtosis calculation
    skewness = skew(embedding_numpy.flatten())
    kurt = kurtosis(embedding_numpy.flatten())
    logger.info(f"Skewness of embeddings: {skewness}")
    logger.info(f"Kurtosis of embeddings: {kurt}")

    # 5. Pairwise Distance (Euclidean)
    # pairwise_distances = pdist(embedding_numpy, 'euclidean')  # Or use 'cosine' distance
    # logger.info(f"Pairwise distance (mean): {np.mean(pairwise_distances)}")

    # 6. Cosine Similarity with True Labels (if labels available)
    if labels is not None:
        # Reshape labels to match embeddings' dimension for cosine similarity
        labels_reshaped = labels.unsqueeze(1).expand_as(embeddings)  # Add a dimension to match
        cos_sim = torch.nn.functional.cosine_similarity(embeddings, labels_reshaped, dim=1)
        logger.info(f"Average Cosine similarity with true labels: {cos_sim.mean().item()}")

    # 7. Outliers Detection (Using Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    outlier_flags = lof.fit_predict(embedding_numpy)  # -1 for outliers, 1 for inliers
    num_outliers = np.sum(outlier_flags == -1)
    logger.info(f"Number of outliers detected in embeddings: {num_outliers}")

    logger.info("Embeddings analysis completed.")

def create_dataloader(embeddings_tensor, labels_tensor, batch_size=128):
    """
    Create a DataLoader for embeddings and their corresponding labels.

    Args:
        embeddings_tensor (torch.Tensor): Tensor of embeddings.
        labels_tensor (torch.Tensor): Tensor of labels.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the provided tensors.
    """
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_embedding_loaders(embeddings_dict, labels_tensor, batch_size=128):
    """
    Creates DataLoaders for each embedding method in the dictionary.

    Args:
        embeddings_dict (dict): Dictionary where keys are method names and values are embedding tensors.
        labels_tensor (torch.Tensor): Tensor of labels corresponding to embeddings.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: Dictionary containing DataLoaders for each embedding method.
    """
    loaders = {}
    for method, embeddings in embeddings_dict.items():
        loaders[method] = create_dataloader(embeddings, labels_tensor, batch_size=batch_size)
    return loaders

def visualize_embeddings(embeddings, labels, title="t-SNE Visualization"):
    """
    Visualizes embeddings using t-SNE.

    Args:
        embeddings (torch.Tensor): Embeddings to visualize.
        labels (torch.Tensor): Corresponding labels.
        title (str): Title of the plot.
    """
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_np, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

def generate_embeddings(model, data_loader, embedding_type, device="cpu"):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, label_batch in data_loader:
            images = images.to(device)
            if embedding_type == "autoencoder":
                encoded, _ = model(images)
            elif embedding_type == "vae":
                mu, _, _ = model(images)
                encoded = mu  # Use the mean of the latent space
            elif embedding_type == "dae":
                _, _, encoded = model(images)
            else:
                raise ValueError(f"Embedding type '{embedding_type}' is not recognized.")

            embeddings.append(encoded.cpu())
            labels.append(label_batch)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels