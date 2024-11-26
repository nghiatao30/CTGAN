# pool_match.py
import os
import torch
import numpy as np
import pandas as pd
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ctgan import CTGAN
from utils import AverageMeter, accuracy
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_tabular_data(data_path):
    data = pd.read_csv(data_path)
    categorical_columns = []
    continuous_columns = [col for col in data.columns if col not in categorical_columns]

    for col in continuous_columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data, categorical_columns, continuous_columns

#def define_discriminator(data_dim):
    return Discriminator(input_dim=data_dim, discriminator_dim=(256, 256), pac=10).cuda()

def calc_gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
    device = real_data.device
    alpha = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data.clone().detach() + (1 - alpha) * fake_data.clone().detach()
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty



def sample_by_label(ctgan, label_counts, label_column, label_mapping):
    """
    Sinh dữ liệu tổng hợp cho mỗi nhãn với số lượng mẫu cụ thể từ dữ liệu gốc (oridata).
    
    Args:
        ctgan (CTGAN): Đối tượng CTGAN đã được huấn luyện.
        label_counts (dict): Số lượng mẫu cho từng nhãn.
        label_column (str): Tên cột nhãn.
        label_mapping (dict): Ánh xạ giữa nhãn gốc và mã hóa.
        
    Returns:
        torch.Tensor: Tensor dữ liệu tổng hợp với số lượng mẫu tương ứng cho từng nhãn.
    """
    synthetic_data = []

    for label, count in label_counts.items():
        # Chuyển nhãn thành kiểu chuỗi trước khi tra cứu
        label_str = int(label)
        if label_str not in label_mapping:
            raise ValueError(f"Nhãn {label} không tồn tại trong mapping.")

        encoded_label = label_mapping[label_str]
        batch_size = ctgan._batch_size  # Sử dụng batch_size từ CTGAN
        total_synthetic = 0
        current_synthetic = []

        while total_synthetic < count:
            remaining = min(batch_size, count - total_synthetic)
            synthetic_samples = ctgan.sample_by_label(
                n=remaining, 
                label_column=label_column, 
                label_value=encoded_label
            )
            synthetic_samples_df = pd.DataFrame(synthetic_samples)
            current_synthetic.append(synthetic_samples_df)
            total_synthetic += len(synthetic_samples_df)

        # Kết hợp các mẫu tổng hợp cho nhãn này
        current_synthetic = pd.concat(current_synthetic, ignore_index=True)

        # Kiểm tra và điều chỉnh số lượng mẫu
        if total_synthetic > count:
            current_synthetic = current_synthetic.iloc[:count]
        elif total_synthetic < count:
            remaining = count - total_synthetic
            additional_samples = ctgan.sample_by_label(
                n=remaining, 
                label_column=label_column, 
                label_value=encoded_label
            )
            additional_samples_df = pd.DataFrame(additional_samples)
            current_synthetic = pd.concat([current_synthetic, additional_samples_df], ignore_index=True)
        
        # Thêm dữ liệu của nhãn này vào danh sách tổng hợp
        synthetic_data.append(current_synthetic)    
    # Kết hợp tất cả các mẫu tổng hợp và chuyển thành Tensor
    synthetic_data = pd.concat(synthetic_data, ignore_index=True)
    synthetic_data = preprocess_data(synthetic_data)
    return torch.tensor(synthetic_data.values, dtype=torch.float32)
def matchloss(args, real_data, fake_data, real_labels, fake_labels, discriminator):
    """
    Computes the matching loss between real and synthetic data.

    Args:
        args: Configuration arguments, may include match coefficients or parameters.
        real_data (torch.Tensor): Real data batch.
        fake_data (torch.Tensor): Fake data batch generated by the generator.
        real_labels (torch.Tensor): Labels corresponding to real data.
        fake_labels (torch.Tensor): Labels corresponding to fake data.
        discriminator: The discriminator model to evaluate the matching.

    Returns:
        torch.Tensor: Calculated matching loss.
    """
    with torch.no_grad():
        # Get embeddings or features from discriminator for real and fake data
        real_features = discriminator(real_data)
        fake_features = discriminator(fake_data)

    # Compute the L2 norm between real and fake features
    match_loss = torch.nn.functional.mse_loss(real_features, fake_features)

    # Apply optional coefficient from args if available
    if hasattr(args, 'match_coeff'):
        match_loss *= args.match_coeff

    return match_loss

def train(args, ctgan, discriminator, optim_d, trainloader, label_mapping, criterion, aug=None, aug_rand=None):
    """
    Hàm huấn luyện với việc tối ưu hóa generator của CTGAN.

    Args:
        args: Tham số cấu hình.
        ctgan (CTGAN): Mô hình CTGAN đã được huấn luyện.
        discriminator: Mô hình Discriminator.
        optim_d: Optimizer cho discriminator.
        trainloader: DataLoader chứa các batch từ real_data.
        label_mapping: Mapping giữa nhãn thực và mã hóa.
        criterion: Loss function để tính toán class loss.
        aug: Augmentation function (nếu có).
        aug_rand: Random augmentation (nếu có).
    """
    torch.autograd.set_detect_anomaly(True)
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    logits_losses = AverageMeter()
    top1 = AverageMeter()

    generator = ctgan._generator  # Lấy generator từ CTGAN
    optim_g = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.5, 0.9))

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        for batch_idx, (real_data,) in enumerate(trainloader):
            real_data = real_data.cuda()
            real_data = torch.tensor(real_data, dtype=torch.float32)
            real_labels = real_data[:, -1].float()  # Giả sử nhãn là cột cuối
            label_counts = dict(pd.Series(real_labels.cpu().numpy()).value_counts())

            # --- Train Generator ---
            optim_g.zero_grad()
            discriminator.eval()  # Ensure discriminator is in evaluation mode

            # Generate synthetic labels and data
            gen_labels = torch.randint(0, len(label_mapping), (args.batch_size,), device=real_data.device)
            gen_labels = gen_labels.long().unsqueeze(1)
            #label_counts = dict(pd.Series(gen_labels.cpu().numpy()).value_counts())
            gen_data = ctgan.sample(len(real_data))  # Generate synthetic data
            gen_data = preprocess_data(gen_data)
            gen_data = torch.tensor(gen_data.values, dtype=torch.float32).cuda()  # Convert to tensor and move to GPU

            gen_data = gen_data.cuda()  # Ensure synthetic data is on GPU

            # Verify `gen_class` and `gen_labels`
            gen_source, gen_class = discriminator.predict_with_logits(gen_data) # Chuyển đổi kích thước của gen_class
            #print(f"gen_class shape: {gen_class.shape}, gen_labels shape: {gen_labels.shape}")
            #print(f"gen_class: {gen_class}")
            ##print(f"gen_class logits range: [{gen_class.min().item()}, {gen_class.max().item()}]")

            gen_labels = gen_labels.float() 
            # Compute Generator Loss
            gen_source_loss = -torch.mean(gen_source)
            gen_class_loss = criterion(gen_class, gen_labels)
            gen_loss = gen_source_loss + gen_class_loss

            if aug and aug_rand:
                aug_real = aug(real_data)
                aug_fake = aug(gen_data)
                match_loss = matchloss(args, aug_real, aug_fake, real_labels, gen_labels, discriminator)
                gen_loss += match_loss

            gen_loss.backward()
            optim_g.step()  # Optimize generator
            gen_losses.update(gen_loss.item(), real_data.size(0))

            # --- Train Discriminator ---
            optim_d.zero_grad()
            discriminator.train()

            with torch.no_grad():
                fake_data = gen_data.clone()

            fake_preds, fake_logits = discriminator.predict_with_logits(fake_data)
            real_preds, real_logits = discriminator.predict_with_logits(real_data)

            acc1 = accuracy(real_logits, real_labels, topk=(1,))[0]
            top1.update(acc1.item(), real_data.size(0))

            fake_source_loss = torch.mean(fake_preds)
            real_source_loss = -torch.mean(real_preds)
            real_logits = real_logits.squeeze(dim=1)
            real_class_loss = criterion(real_logits, real_labels)
            fake_class_loss = criterion(fake_logits, gen_labels)

            gradient_penalty = calc_gradient_penalty(discriminator, real_data, fake_data)
            disc_loss = real_source_loss + fake_source_loss + real_class_loss + fake_class_loss + gradient_penalty

            disc_loss.backward()
            optim_d.step()  # Tối ưu hóa discriminator
            disc_losses.update(disc_loss.item(), real_data.size(0))

            # --- Logits Matching Loss ---
            logits_loss = nn.MSELoss()(real_logits, fake_logits)
            logits_losses.update(logits_loss.item(), real_data.size(0))

            # --- Print Progress ---
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'[Epoch {epoch} Iter {batch_idx + 1}] G Loss: {gen_losses.val:.3f} ({gen_losses.avg:.3f}) | '
                      f'D Loss: {disc_losses.val:.3f} ({disc_losses.avg:.3f}) | '
                      f'Logits Loss: {logits_losses.val:.3f} ({logits_losses.avg:.3f}) | '
                      f'D Acc: {top1.val:.3f} ({top1.avg:.3f})')

        # --- Save Model Each Epoch ---
        model_save_path_g = os.path.join(args.output_dir, f'generator_epoch_{epoch}.pth')
        model_save_path_d = os.path.join(args.output_dir, f'discriminator_epoch_{epoch}.pth')
        torch.save(generator.state_dict(), model_save_path_g)
        torch.save(discriminator.state_dict(), model_save_path_d)
        print(f'Models saved at {model_save_path_g} and {model_save_path_d} for epoch {epoch}')



def preprocess_data(data):
    data = data.copy()  # Tạo bản sao để tránh thay đổi dữ liệu gốc
    
    # Lấy danh sách các cột numeric và categorical
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    # Chuẩn hóa các cột số
    if len(numerical_columns) > 0:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # Mã hóa các cột danh mục
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
    
    return data

