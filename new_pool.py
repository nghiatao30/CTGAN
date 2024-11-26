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

def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

    if 'feat' in args.match:
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric) * 0.001)

    elif 'grad' in args.match:
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric) * 0.001)

    elif 'logit' in args.match:
        output_real = F.log_softmax(model(img_real), dim=1)
        output_syn = F.log_softmax(model(img_syn), dim=1)
        loss = add_loss(loss, ((output_real - output_syn) ** 2).mean() * 0.01)

    return loss

def train_match_model(args, model, optim_model, trainloader, criterion, aug_rand):
    '''The training function for the match model
    '''
    for batch_idx, (img, lab) in enumerate(trainloader):
        if batch_idx == args.epochs_match_train:
            break

        img = img.cuda()
        lab = lab.cuda()

        output = model(aug_rand(img))
        loss = criterion(output, lab)

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()



if __name__ == '__main__':
    dataset_path = 'iotid20.csv'
    output_dir = 'new_outputs'
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

    model_dict = torch.load("new_outputs\generator_epoch200.pth")
    generator.load_state_dict(model_dict['generator'])
    discriminator.load_state_dict(model_dict['discriminator'])
    optimizerG.load_state_dict(model_dict['optim_g'])
    optimizerD.load_state_dict(model_dict['optim_d'])

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
                    f'pool_outputs\model_epoch{epoch+1}.pth')



