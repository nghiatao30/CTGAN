import torch
import numpy as np

class TabularTransform:
    def __init__(self, means, stds):
        """
        Transform for normalizing tabular data.
        
        Parameters:
        - means: A list of means for each feature in the tabular data.
        - stds: A list of standard deviations for each feature in the tabular data.
        """
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def __call__(self, x):
        """
        Apply normalization to the input tensor.
        
        Parameters:
        - x: A tensor representing a batch of tabular data.
        
        Returns:
        - A tensor of the same shape as x, with normalization applied.
        """
        return (x - self.means) / self.stds

    def inverse_transform(self, x):
        """
        Revert the normalization to obtain the original data values.
        
        Parameters:
        - x: A normalized tensor.
        
        Returns:
        - A tensor of the same shape as x, reverted from normalization.
        """
        return (x * self.stds) + self.means

