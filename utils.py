
import torch
import numpy as np
import os
import sys
import time

# Assuming TabularTransform is defined here or imported
class TabularTransform:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def __call__(self, x):
        return (x - self.means) / self.stds

# Modify other utility functions as needed
def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Other utility functions related to image-specific operations can be removed or adapted for tabular data

if __name__ == "__main__":
    # Demonstration of using TabularTransform
    means = [0.5] * 20  # Example means for each feature
    stds = [0.2] * 20  # Example standard deviations for each feature
    transform = TabularTransform(means, stds)
    
    # Example tabular data
    data = torch.randn(32, 20)  # Batch size of 32, 20 features
    normalized_data = transform(data)
    print("Normalized data shape:", normalized_data.shape)


