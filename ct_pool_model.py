import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# MLP Model (Multi-Layer Perceptron)
class MLP(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ResNet18 Model (modified for tabular data)
class ResNet18(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNet18, self).__init__()
        # We would typically use pre-trained ResNet18 here, but let's use a basic MLP-based structure
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# TabNet Model (Transformer-based model for tabular data)

# WideNet Model (Wide and Deep Learning approach)
class WideNet(nn.Module):
    def __init__(self, num_classes):
        super(WideNet, self).__init__()
        # Simple wide + deep model structure for tabular data
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DNN Model (Deep Neural Network)
class DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Autoencoder Model for Tabular Data
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TabNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TabNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Sử dụng các layers chuẩn của TabNet (có thể cần các thư viện ngoài để cài đặt đầy đủ)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MLP_Deep(nn.Module):
    """
    A deeper Multi-layer Perceptron for tabular data.
    """
    def __init__(self, input_dim, num_classes):
        super(MLP_Deep, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNetTabular(nn.Module):
    """
    A simple ResNet-like architecture for tabular data (simplified for tabular data).
    """
    def __init__(self, input_dim, num_classes):
        super(ResNetTabular, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Linear(input_dim, 512)  # Shortcut connection
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x + shortcut  # Adding the shortcut connection
        return x

class DenseNetTabular(nn.Module):
    """
    A simple DenseNet-like architecture for tabular data (simplified for tabular data).
    """
    def __init__(self, input_dim, num_classes):
        super(DenseNetTabular, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x