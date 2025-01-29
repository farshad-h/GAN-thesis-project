# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
# from plan2_gan_models import (
#     SimpleGANGenerator, SimpleGANDiscriminator,
#     ContrastiveGANGenerator, ContrastiveGANDiscriminator,
#     VAEGANEncoder, VAEGANGenerator, VAEGANDiscriminator,
#     WGANGenerator, WGANCritic,
#     CrossDomainGenerator, CrossDomainDiscriminator,
#     CycleGenerator,
#     DualGANGenerator, DualGANDiscriminator,
#     ContrastiveDualGANGenerator, ContrastiveDualGANDiscriminator,
#     compute_gradient_penalty
# )

# Gradient Penalty for WGAN-GP
def compute_gradient_penalty(model, real_samples, fake_samples, device):
    """
    Computes the gradient penalty for WGAN-GP.

    The gradient penalty is a regularization term added to the WGAN-GP loss function 
    to enforce the Lipschitz constraint. This penalty ensures that the gradient of 
    the critic's output with respect to its input is close to 1.

    Args:
        model (nn.Module): The critic or discriminator model used in WGAN-GP.
        real_samples (torch.Tensor): A batch of real data samples.
        fake_samples (torch.Tensor): A batch of generated (fake) data samples.
        device (torch.device): The device (CPU or GPU) to run the computation on.

    Returns:
        torch.Tensor: The computed gradient penalty value.
    """
    # Get the batch size from the real samples
    batch_size = real_samples.size(0)

    # Create random interpolation coefficients (alpha)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)

    # Perform linear interpolation between real and fake samples
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Forward pass through the model (critic) to get predictions
    interpolated_pred = model(interpolated)

    # Create a tensor of ones with the same shape as the model output for computing gradients
    grad_outputs = torch.ones(interpolated_pred.size(), device=device)

    # Compute gradients w.r.t. interpolated samples
    grad_interpolated = torch.autograd.grad(
        outputs=interpolated_pred,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the L2 norm of the gradient for each sample
    grad_norm = grad_interpolated.view(batch_size, -1).norm(2, dim=1)

    # Compute the gradient penalty (difference between the norm and 1)
    gradient_penalty = torch.mean((grad_norm - 1) ** 2)

    return gradient_penalty

# Training WGAN-GP
def train_wgan_gp(generator, critic, **kwargs):
    """
    Trains WGAN-GP with gradient penalty.

    Args:
        generator (nn.Module): WGAN Generator model.
        critic (nn.Module): WGAN Critic model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'lambda_gp', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    lambda_gp = kwargs.get('lambda_gp', 10)  # Default value for lambda_gp (only for WGAN-GP)
    learning_rate = kwargs.get('learning_rate', 0.0001)  # Default learning rate

    # Optimizers for Generator and Critic
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings in embedding_loader:
            batch_size = real_embeddings.size(0)

            # Train Critic
            optimizer_c.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = generator(noise)

            real_validity = critic(real_embeddings)
            fake_validity = critic(fake_embeddings.detach())
            gp = compute_gradient_penalty(critic, real_embeddings, fake_embeddings, device)
            loss_c = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            loss_c.backward()
            optimizer_c.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_embeddings = generator(noise)
            fake_validity = critic(fake_embeddings)
            loss_g = -torch.mean(fake_validity)

            loss_g.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss Critic: {loss_c.item():.4f}, Loss Generator: {loss_g.item():.4f}")

# Training VAE-GAN
def train_vae_gan(encoder, generator, discriminator, **kwargs):
    """
    Trains VAE-GAN with adversarial and reconstruction loss.

    Args:
        encoder (nn.Module): VAE-GAN Encoder model.
        generator (nn.Module): VAE-GAN Generator model.
        discriminator (nn.Module): VAE-GAN Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    optimizer_g = optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss for adversarial loss
    mse_loss = nn.MSELoss()  # Mean Squared Error for reconstruction loss

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            mu, logvar = encoder(real_embeddings)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # Reparameterization trick
            fake_embeddings = generator(z)

            real_validity = discriminator(real_embeddings)
            fake_validity = discriminator(fake_embeddings.detach())
            d_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                     bce_loss(fake_validity, torch.zeros_like(fake_validity))

            d_loss.backward()
            optimizer_d.step()

            # Train Generator and Encoder
            optimizer_g.zero_grad()
            fake_validity = discriminator(fake_embeddings)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence Loss
            recon_loss = mse_loss(fake_embeddings, real_embeddings)  # Reconstruction Loss
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity)) + kl_loss + recon_loss

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


# Training Contrastive GAN
def train_contrastive_gan(generator, discriminator, **kwargs):
    """
    Trains Contrastive GAN with adversarial and contrastive loss.

    Args:
        generator (nn.Module): Contrastive GAN Generator model.
        discriminator (nn.Module): Contrastive GAN Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    # Optimizers for Generator and Discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss functions
    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    contrastive_loss_fn = nn.CrossEntropyLoss()  # Contrastive Loss

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = generator(noise)

            real_validity = discriminator(real_embeddings)
            fake_validity = discriminator(fake_embeddings.detach())
            d_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                     bce_loss(fake_validity, torch.zeros_like(fake_validity))

            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity = discriminator(fake_embeddings)
            c_loss = contrastive_loss_fn(fake_embeddings, real_embeddings)
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity)) + c_loss

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Training Cross-Domain GAN
def train_cross_domain_gan(generator, discriminator, **kwargs):
    """
    Trains Cross-Domain GAN with adversarial loss.

    Args:
        generator (nn.Module): Cross-Domain Generator model.
        discriminator (nn.Module): Cross-Domain Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = generator(noise)

            real_validity = discriminator(real_embeddings)
            fake_validity = discriminator(fake_embeddings.detach())
            d_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                     bce_loss(fake_validity, torch.zeros_like(fake_validity))

            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity = discriminator(fake_embeddings)
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity))

            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Training CycleGAN
def train_cycle_gan(generator_a, generator_b, discriminator_a, discriminator_b, **kwargs):
    """
    Trains CycleGAN with cycle consistency and adversarial losses.

    Args:
        generator_a (nn.Module): Generator for domain A → B.
        generator_b (nn.Module): Generator for domain B → A.
        discriminator_a (nn.Module): Discriminator for domain A.
        discriminator_b (nn.Module): Discriminator for domain B.
        **kwargs: Additional arguments including 'data_loader_a', 'data_loader_b', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader_a = kwargs.get('data_loader_a')
    embedding_loader_b = kwargs.get('data_loader_b')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    # Optimizers for Generators and Discriminators
    optimizer_g = optim.Adam(list(generator_a.parameters()) + list(generator_b.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_a = optim.Adam(discriminator_a.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_b = optim.Adam(discriminator_b.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss functions
    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    cycle_loss_fn = nn.L1Loss()  # Cycle consistency loss (L1 loss)

    # Training Loop
    for epoch in range(epochs):
        for real_a, real_b in zip(embedding_loader_a, embedding_loader_b):
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            # Train Discriminators
            optimizer_d_a.zero_grad()
            optimizer_d_b.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            real_validity_a = discriminator_a(real_a)
            real_validity_b = discriminator_b(real_b)

            fake_validity_a = discriminator_a(fake_a.detach())
            fake_validity_b = discriminator_b(fake_b.detach())

            d_loss_a = bce_loss(real_validity_a, torch.ones_like(real_validity_a)) + \
                       bce_loss(fake_validity_a, torch.zeros_like(fake_validity_a))

            d_loss_b = bce_loss(real_validity_b, torch.ones_like(real_validity_b)) + \
                       bce_loss(fake_validity_b, torch.zeros_like(fake_validity_b))

            d_loss_a.backward()
            d_loss_b.backward()

            optimizer_d_a.step()
            optimizer_d_b.step()

            # Train Generators
            optimizer_g.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            cycle_a = generator_b(fake_b)
            cycle_b = generator_a(fake_a)

            cycle_loss = cycle_loss_fn(cycle_a, real_a) + cycle_loss_fn(cycle_b, real_b)
            g_loss_a = bce_loss(discriminator_b(fake_b), torch.ones_like(fake_validity_b)) + cycle_loss
            g_loss_b = bce_loss(discriminator_a(fake_a), torch.ones_like(fake_validity_a)) + cycle_loss

            g_loss = g_loss_a + g_loss_b

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss A: {d_loss_a.item():.4f}, D Loss B: {d_loss_b.item():.4f}, G Loss: {g_loss.item():.4f}")

# Training Dual GAN
def train_dual_gan(generator_a, generator_b, discriminator_a, discriminator_b, **kwargs):
    """
    Trains Dual GAN with identity and adversarial losses.

    Args:
        generator_a (nn.Module): Generator for domain A → B.
        generator_b (nn.Module): Generator for domain B → A.
        discriminator_a (nn.Module): Discriminator for domain A.
        discriminator_b (nn.Module): Discriminator for domain B.
        **kwargs: Additional arguments including 'data_loader_a', 'data_loader_b', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader_a = kwargs.get('data_loader_a')
    embedding_loader_b = kwargs.get('data_loader_b')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    # Optimizers for Generators and Discriminators
    optimizer_g = optim.Adam(list(generator_a.parameters()) + list(generator_b.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_a = optim.Adam(discriminator_a.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_b = optim.Adam(discriminator_b.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss functions
    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    identity_loss_fn = nn.L1Loss()  # Identity loss (for cycle consistency)

    # Training Loop
    for epoch in range(epochs):
        for real_a, real_b in zip(embedding_loader_a, embedding_loader_b):
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            # Train Discriminators
            optimizer_d_a.zero_grad()
            optimizer_d_b.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            real_validity_a = discriminator_a(real_a)
            real_validity_b = discriminator_b(real_b)

            fake_validity_a = discriminator_a(fake_a.detach())
            fake_validity_b = discriminator_b(fake_b.detach())

            d_loss_a = bce_loss(real_validity_a, torch.ones_like(real_validity_a)) + \
                       bce_loss(fake_validity_a, torch.zeros_like(fake_validity_a))

            d_loss_b = bce_loss(real_validity_b, torch.ones_like(real_validity_b)) + \
                       bce_loss(fake_validity_b, torch.zeros_like(fake_validity_b))

            d_loss_a.backward()
            d_loss_b.backward()

            optimizer_d_a.step()
            optimizer_d_b.step()

            # Train Generators
            optimizer_g.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            identity_a = generator_b(real_a)
            identity_b = generator_a(real_b)

            identity_loss = identity_loss_fn(identity_a, real_a) + identity_loss_fn(identity_b, real_b)
            g_loss_a = bce_loss(discriminator_b(fake_b), torch.ones_like(fake_validity_b)) + identity_loss
            g_loss_b = bce_loss(discriminator_a(fake_a), torch.ones_like(fake_validity_a)) + identity_loss

            g_loss = g_loss_a + g_loss_b

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss A: {d_loss_a.item():.4f}, D Loss B: {d_loss_b.item():.4f}, G Loss: {g_loss.item():.4f}")

# Training Contrastive-Guided Dual GAN
def train_contrastive_dual_gan(generator_a, generator_b, discriminator_a, discriminator_b, **kwargs):
    """
    Trains Contrastive-Guided Dual GAN with adversarial and contrastive losses.

    Args:
        generator_a (nn.Module): Generator for domain A → B.
        generator_b (nn.Module): Generator for domain B → A.
        discriminator_a (nn.Module): Discriminator for domain A.
        discriminator_b (nn.Module): Discriminator for domain B.
        **kwargs: Additional arguments including 'data_loader_a', 'data_loader_b', 'latent_dim', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader_a = kwargs.get('data_loader_a')
    embedding_loader_b = kwargs.get('data_loader_b')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    # Optimizers for Generators and Discriminators
    optimizer_g = optim.Adam(list(generator_a.parameters()) + list(generator_b.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_a = optim.Adam(discriminator_a.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d_b = optim.Adam(discriminator_b.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss functions
    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    contrastive_loss_fn = nn.CrossEntropyLoss()  # Contrastive Loss

    # Training Loop
    for epoch in range(epochs):
        for real_a, real_b in zip(embedding_loader_a, embedding_loader_b):
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            # Train Discriminators
            optimizer_d_a.zero_grad()
            optimizer_d_b.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            real_validity_a = discriminator_a(real_a)
            real_validity_b = discriminator_b(real_b)

            fake_validity_a = discriminator_a(fake_a.detach())
            fake_validity_b = discriminator_b(fake_b.detach())

            d_loss_a = bce_loss(real_validity_a, torch.ones_like(real_validity_a)) + \
                       bce_loss(fake_validity_a, torch.zeros_like(fake_validity_a))

            d_loss_b = bce_loss(real_validity_b, torch.ones_like(real_validity_b)) + \
                       bce_loss(fake_validity_b, torch.zeros_like(fake_validity_b))

            d_loss_a.backward()
            d_loss_b.backward()

            optimizer_d_a.step()
            optimizer_d_b.step()

            # Train Generators
            optimizer_g.zero_grad()

            fake_b = generator_a(real_a)
            fake_a = generator_b(real_b)

            c_loss_a = contrastive_loss_fn(fake_b, real_b)
            c_loss_b = contrastive_loss_fn(fake_a, real_a)

            g_loss_a = bce_loss(discriminator_b(fake_b), torch.ones_like(fake_validity_b)) + c_loss_a
            g_loss_b = bce_loss(discriminator_a(fake_a), torch.ones_like(fake_validity_a)) + c_loss_b

            g_loss = g_loss_a + g_loss_b

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss A: {d_loss_a.item():.4f}, D Loss B: {d_loss_b.item():.4f}, G Loss: {g_loss.item():.4f}")

# Training Semi-Supervised GAN
def train_semi_supervised_gan(generator, discriminator, **kwargs):
    """
    Trains Semi-Supervised GAN with classification loss and adversarial loss.

    Args:
        generator (nn.Module): Semi-Supervised GAN Generator model.
        discriminator (nn.Module): Semi-Supervised GAN Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'num_classes', 'epochs', 'device', 'learning_rate'.
    """
    # Extract arguments from kwargs
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    num_classes = kwargs.get('num_classes')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)  # Default learning rate

    # Optimizers for Generator and Discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss functions
    bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    class_loss = nn.CrossEntropyLoss()  # Cross-Entropy Loss for classification

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings, labels in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            labels = labels.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = generator(noise)

            real_validity, real_classes = discriminator(real_embeddings)
            fake_validity, _ = discriminator(fake_embeddings.detach())

            adv_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                       bce_loss(fake_validity, torch.zeros_like(fake_validity))
            class_loss_real = class_loss(real_classes, labels)

            d_loss = adv_loss + class_loss_real
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity, fake_classes = discriminator(fake_embeddings)
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity)) + \
                     class_loss(fake_classes, labels)

            g_loss.backward()
            optimizer_g.step()

        # Print losses for each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

def train_conditional_gan(generator, discriminator, **kwargs):
    """
    Trains Conditional GAN (cGAN) with adversarial loss conditioned on labels.

    Args:
        generator (nn.Module): Conditional GAN Generator model.
        discriminator (nn.Module): Conditional GAN Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'learning_rate', 'num_classes'.
    """
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)
    num_classes = kwargs.get('num_classes')

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings, labels in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            labels = labels.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = generator(noise, labels)

            real_validity = discriminator(real_embeddings, labels)
            fake_validity = discriminator(fake_embeddings.detach(), labels)
            d_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                     bce_loss(fake_validity, torch.zeros_like(fake_validity))

            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity = discriminator(fake_embeddings, labels)
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity))

            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

def train_infogan(generator, discriminator, **kwargs):
    """
    Trains InfoGAN with both adversarial and information maximization losses.

    Args:
        generator (nn.Module): InfoGAN Generator model.
        discriminator (nn.Module): InfoGAN Discriminator model.
        **kwargs: Additional arguments including 'data_loader', 'latent_dim', 'epochs', 'device', 'learning_rate', 'num_classes'.
    """
    embedding_loader = kwargs.get('data_loader')
    latent_dim = kwargs.get('latent_dim')
    categorical_dim = kwargs.get('categorical_dim')
    epochs = kwargs.get('epochs')
    device = kwargs.get('device')
    learning_rate = kwargs.get('learning_rate', 0.0002)
    num_classes = kwargs.get('num_classes')

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(epochs):
        for real_embeddings, labels in embedding_loader:
            real_embeddings = real_embeddings.to(device)
            labels = labels.to(device)
            batch_size = real_embeddings.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            c = torch.randint(0, num_classes, (batch_size,), device=device)  # Random categorical labels
            fake_embeddings = generator(noise, c)

            real_validity, real_class = discriminator(real_embeddings)
            fake_validity, fake_class = discriminator(fake_embeddings.detach())

            d_loss = bce_loss(real_validity, torch.ones_like(real_validity)) + \
                     bce_loss(fake_validity, torch.zeros_like(fake_validity)) + \
                     ce_loss(real_class, labels)  # Add categorical loss

            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity, fake_class = discriminator(fake_embeddings)
            g_loss = bce_loss(fake_validity, torch.ones_like(fake_validity)) + \
                     ce_loss(fake_class, c)  # Maximize mutual information

            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
