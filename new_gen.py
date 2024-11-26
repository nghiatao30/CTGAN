import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
import numpy as np
# Assuming gan_model.py is adjusted as discussed earlier and located in the same directory
from newgan_model import Generator, Discriminator

# Placeholder for the real path to your dataset
dataset_path = 'iotid20.csv'
output_dir = 'new_outputs'
# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Handling NaN values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column].fillna(df[column].mean(), inplace=True)
    
    # Handling NaN values for non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for column in non_numeric_columns:
        df[column].fillna('Unknown', inplace=True)  # Example placeholder value

    X = df.iloc[:, :-1]
    
    # Convert categorical columns to numeric if needed
    le = LabelEncoder()
    for column in non_numeric_columns:
        if column in X.columns:  # Check if the column is in features
            X[column] = le.fit_transform(X[column])

    # Using QuantileTransformer for scaling
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    X_normalized = qt.fit_transform(X.astype(float))

    # Assuming the last column is the target
    y = df.iloc[:, -1].values
    # Encoding the target variable if it's categorical
    y = le.fit_transform(y)
    
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

# Adjust the path to point to your actual dataset
dataset = load_and_preprocess_data(dataset_path)

# Now that the dataset is loaded, we can determine the number of features
num_features = dataset.tensors[0].shape[1]  # Assuming all features are used

# Function to split the dataset into training and validation sets
def split_dataset(dataset, test_size=0.2, batch_size=256):
    # train_dataset, val_dataset = train_test_split(dataset, test_size=test_size, random_state=42)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader

# Define Generator and Discriminator with correct dimensions
input_dim = 79  # Dimensionality of the noise vector
generator = Generator(input_dim=input_dim, output_dim=num_features).cuda()
discriminator = Discriminator(input_dim=num_features).cuda()

# Training parameters
lr = 0.0002
beta1 = 0.5
num_epochs = 200

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Load and preprocess data
train_loader = split_dataset(dataset)

# Lists to keep track of progress
G_losses = []
D_losses = []

print("Starting Training Loop...")
# For each epoch
save_csv_flag = False

for epoch in range(num_epochs):
    for i, (data, _) in enumerate(train_loader, 0):
        # Huấn luyện Discriminator
        discriminator.zero_grad()
        real_data = data.cuda()
        b_size = real_data.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device='cuda')
        output = discriminator(real_data).view(-1)
        errD_real = nn.BCELoss()(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, input_dim, device='cuda')
        fake_data = generator(noise)
        label.fill_(0)
        output = discriminator(fake_data.detach()).view(-1)
        errD_fake = nn.BCELoss()(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Huấn luyện Generator
        generator.zero_grad()
        label.fill_(1)
        output = discriminator(fake_data).view(-1)
        errG = nn.BCELoss()(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # Lưu trọng số sau mỗi epoch
    if (epoch + 1) % 10 == 0:
        model_dict = {'generator': generator.state_dict(),
                          'discriminator': discriminator.state_dict(),
                          'optim_g': optimizerG.state_dict(),
                          'optim_d': optimizerD.state_dict()}
        torch.save(
                model_dict,
                f'new_outputs\model_epoch{epoch+1}.pth')



