import os
import torch
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from ctgan import CTGAN, Discriminator
from utils import AverageMeter

# Load and preprocess the tabular data
def load_tabular_data(data_path):
    data = pd.read_csv(data_path)
    categorical_columns = []  # Define categorical columns based on data
    continuous_columns = [col for col in data.columns if col not in categorical_columns]

    # Normalize continuous columns
    for col in continuous_columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data, categorical_columns, continuous_columns

# Training function for CTGAN-style discriminator
def train(args, ctgan, discriminator, optim_d, trainloader):
    disc_losses = AverageMeter()

    for epoch in range(args.epochs):
        discriminator.train()

        for batch_idx, (img_real,) in enumerate(trainloader):
            img_real = img_real.cuda()

            # Step 1: Discriminator training step
            optim_d.zero_grad()

            # Generate synthetic samples from CTGAN
            img_syn = torch.tensor(ctgan.sample(len(img_real)), dtype=torch.float32).cuda()

            # Compute discriminator outputs for real and synthetic data
            real_preds = discriminator(img_real)
            fake_preds = discriminator(img_syn)

            # Discriminator loss calculation
            disc_loss = -torch.mean(torch.log(real_preds) + torch.log(1 - fake_preds))
            disc_loss.backward()
            optim_d.step()
            disc_losses.update(disc_loss.item())

            # Print progress every 'print_freq' batches
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'[Train Epoch {epoch} Iter {batch_idx + 1}] D Loss: {disc_losses.val:.3f}({disc_losses.avg:.3f})')

# Main execution
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--data-path', type=str, required=True)
    args = parser.parse_args()

    # Load and prepare data
    data, categorical_columns, continuous_columns = load_tabular_data(args.data_path)
    train_data = TensorDataset(torch.tensor(data.values, dtype=torch.float32))
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize and fit CTGAN
    ctgan = CTGAN(epochs=100)
    ctgan.fit(data, categorical_columns)

    # Initialize the CTGAN-style discriminator directly
    discriminator = Discriminator(input_dim=data.shape[1], discriminator_dim=(256, 256), pac=10).cuda()

    # Optimizer for discriminator
    optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the discriminator with synthetic data from CTGAN
    train(args, ctgan, discriminator, optim_d, trainloader)
