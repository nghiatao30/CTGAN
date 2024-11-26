import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from newgan_model import Generator, Discriminator  # Ensure this is correctly imported
from utils import accuracy, AverageMeter  # Assuming these are adapted for tabular data

# Placeholder for TabularTransform
class TabularTransform:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def __call__(self, x):
        return (x - self.means) / self.stds

def load_tabular_data(path, batch_size):
    # Load your tabular dataset
    df = pd.read_csv(path)
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Normalize features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create a dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def validate(model, dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc = accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accs.update(acc[0], inputs.size(0))

    print(f'Validation Loss: {losses.avg:.4f}, Accuracy: {accs.avg:.2f}%')

