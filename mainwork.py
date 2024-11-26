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
from ctgan import CTGAN  # Thêm CTGAN
from newgan_model import Generator, Discriminator  # Các mô hình đã chỉnh sửa

# Đường dẫn tới dataset
dataset_path = 'iotid20.csv'
output_dir = 'new_outputs'

# Hàm chưng cất dữ liệu bằng DiM
def dim_distillation(data):
    # Thực hiện chưng cất đặc trưng, loại bỏ các mẫu nhiễu không cần thiết
    # Giả sử dữ liệu được chưng cất trở lại dưới dạng pandas DataFrame
    distilled_data = data[data['feature_importance'] > 0.5]  # Ví dụ chưng cất dựa trên mức độ quan trọng
    return distilled_data

# Hàm sinh dữ liệu tổng hợp bằng CTGAN
def generate_synthetic_data_with_ctgan(data, num_samples=1000):
    ctgan = CTGAN(epochs=100)
    ctgan.fit(data)
    synthetic_data = ctgan.sample(num_samples)
    return synthetic_data

# Hàm tải và tiền xử lý dữ liệu
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column].fillna(df[column].mean(), inplace=True)
    return df

# Hàm chính
def main():
    # Bước 1: Tải dữ liệu và chưng cất bằng DiM
    data = load_and_preprocess_data(dataset_path)
    distilled_data = dim_distillation(data)
    
    # Bước 2: Sinh dữ liệu tổng hợp bằng CTGAN
    synthetic_data = generate_synthetic_data_with_ctgan(distilled_data, num_samples=1000)
    
    # Lưu dữ liệu tổng hợp
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    synthetic_data.to_csv(os.path.join(output_dir, "synthetic_data.csv"), index=False)
    print("Synthetic data saved to synthetic_data.csv")

if __name__ == "__main__":
    main()
