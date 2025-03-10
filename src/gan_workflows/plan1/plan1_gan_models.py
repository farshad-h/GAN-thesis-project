# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Generator Block
class G_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, strides=2, padding=2):
        super(G_block, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv2d_trans(x)))

# Simple Generator
class SimpleGenerator(nn.Module):
    def __init__(self, input_dim):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            G_block(256, 128, 5, 1, 2),
            G_block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

# Embedding Generator
class EmbeddingAsInputGenerator(nn.Module):
    def __init__(self, latent_dim, embedding_dim):
        super(EmbeddingAsInputGenerator, self).__init__()
        self.embedding_transform = nn.Linear(embedding_dim, 7 * 7 * 256, bias=False)
        self.model = nn.Sequential(
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            G_block(256, 128, 5, 1, 2),
            G_block(128, 64, 5, 2, 2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, embedding):
        transformed_embedding = self.embedding_transform(embedding)
        transformed_embedding = transformed_embedding.view(z.size(0), 256, 7, 7)
        combined_input = z.unsqueeze(-1).unsqueeze(-1) + transformed_embedding
        return self.model(combined_input)

# Discriminator Block
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1, alpha=0.2, dropout=0.3):
        super(D_block, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding, bias=False
        )
        self.activation = nn.LeakyReLU(alpha, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.conv2d(x)))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            D_block(1, 64, 5, 2, 2),
            D_block(64, 128, 5, 2, 2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# ACGAN Generator
class AC_Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(AC_Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.linear = nn.Linear(latent_dim * 2, 7 * 7 * 256)
        self.bn = nn.BatchNorm1d(7 * 7 * 256)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.unflatten = nn.Unflatten(1, (256, 7, 7))
        self.blocks = nn.Sequential(
            G_block(256, 128, 5, 1, 2),
            G_block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        x = self.linear(gen_input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.unflatten(x)
        return self.blocks(x)

# ACGAN Discriminator
class AC_Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(AC_Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            D_block(1, 64, 5, 2, 2),
            D_block(64, 128, 5, 2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.aux_classifier = nn.Linear(128 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        validity = torch.sigmoid(self.fc(x))
        label = self.aux_classifier(x)
        return validity, label

# InfoGAN Generator
class Info_Generator(nn.Module):
    def __init__(self, embedding_dim, continuous_dim, discrete_dim):
        super(Info_Generator, self).__init__()
        input_dim = embedding_dim + continuous_dim + discrete_dim
        self.linear = nn.Linear(input_dim, 7 * 7 * 256)
        self.bn = nn.BatchNorm1d(7 * 7 * 256)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.unflatten = nn.Unflatten(1, (256, 7, 7))
        self.blocks = nn.Sequential(
            G_block(256, 128, 5, 1, 2),
            G_block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, embedding_with_noise, continuous_code, discrete_code):
        gen_input = torch.cat((embedding_with_noise, continuous_code, discrete_code), dim=-1)
        x = self.linear(gen_input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.unflatten(x)
        return self.blocks(x)

# InfoGAN Discriminator
class Info_Discriminator(nn.Module):
    def __init__(self, continuous_dim, discrete_dim):
        super(Info_Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            D_block(1, 64, 5, 2, 2),
            D_block(64, 128, 5, 2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.q_net = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.q_continuous = nn.Linear(128, continuous_dim)
        self.q_discrete = nn.Linear(128, discrete_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        validity = torch.sigmoid(self.fc(x))
        q_shared = self.q_net(x)
        q_continuous = self.q_continuous(q_shared)
        q_discrete = self.q_discrete(q_shared)
        return validity, q_continuous, q_discrete