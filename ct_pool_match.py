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
import torch.nn.functional as F
from ct_pool_model import *
import random
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

def dist(x, y, method='mse'):
    """ Distance objectives (for tabular data)
    """
    dist_ = None
    # Flatten the tensors if they are 2D (useful for tabular data)
    if x.dim() > 1:
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

    if method == 'mse':
        # Mean Squared Error: sum of squared differences
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        # L1 norm: sum of absolute differences
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        # Mean of L1 distance per sample
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        # Cosine similarity: similarity of the vector representations
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_

def add_loss(loss_sum, loss):

    if loss_sum is None:
        return loss  
    else:
        return loss_sum + loss  




import torch
import torch.nn as nn
import torch.nn.functional as F

def matchloss(args, data_real, data_syn, lab_real, lab_syn, model):
    """Matching losses (feature, gradient or logit matching) for tabular data"""
    loss = None

    # Feature matching for tabular data
    if 'feat' in args.match:
        with torch.no_grad():
            feat_tg = model.get_feature(data_real)  # Change to model's feature extraction
        feat = model.get_feature(data_syn)  # Same for synthetic data

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric) * 0.001)

    # Gradient matching for tabular data
    elif 'grad' in args.match:
        criterion = nn.CrossEntropyLoss()  # Or use MSELoss for regression tasks
        
        # Calculate loss and gradients for real data
        output_real = model(data_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))  # Detach to avoid backprop

        # Calculate loss and gradients for synthetic data
        output_syn = model(data_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        # Compare gradients between real and synthetic data
        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # Ignore bias layers if specified
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:  # Ignore fully connected layers if specified
                continue

            # Add gradient loss (using dist function to measure distance between gradients)
            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric) * 0.001)

    # Logit matching for tabular data
    elif 'logit' in args.match:
        output_real = model(data_real)  # Direct output for tabular data
        output_syn = model(data_syn)    # Direct output for synthetic data
        loss = add_loss(loss, ((output_real - output_syn) ** 2).mean() * 0.01)

    return loss


def define_model(args, num_classes):
    model_pool = [
        'mlp', 'tabnet', 'dnn'
        , 'autoencoder', 'mlp_deep', 'densenet',
    ]   
    
    model_name = random.choice(model_pool)
    print('Random model selected: {}'.format(model_name))

    if model_name == 'mlp':
        return MLP(num_classes, input_dim=args.input_dim)
    elif model_name == 'resnet18':
        return ResNet18(input_dim=args.input_dim, num_classes=num_classes)
    elif model_name == 'tabnet':
        return TabNet(input_dim=args.input_dim, num_classes=num_classes)
    elif model_name == 'widenet':
        return WideNet(num_classes)
    elif model_name == 'dnn':
        return DNN(input_dim=args.input_dim, num_classes=num_classes)
    elif model_name == 'autoencoder':
        return AutoEncoder(input_dim=args.input_dim)
    elif model_name == 'mlp_deep':
        return MLP_Deep(args.input_dim, num_classes)
    elif model_name == 'resnet':
        return ResNetTabular(args.input_dim, num_classes)
    elif model_name == 'densenet':
        return DenseNetTabular(args.input_dim, num_classes)
    return None

# Thêm hàm optim_model vào ct_pool_match.py
def op_model(model, lr, momentum, weight_decay):
    """
    Define optimizer for the match model.
    """
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Thêm hàm train_match_model vào ct_pool_match.py
def train_match_model(args, model, optim_model, trainloader, criterion):
    """
    Training function for the match model.
    """
    model.train()
    for batch_idx, (real_data,) in enumerate(trainloader):
        if batch_idx == args.epochs_match_train:
            break

        if not isinstance(real_data, torch.Tensor):
            real_data = torch.tensor(real_data, dtype=torch.float32)  # Chuyển thành tensor nếu cần

        real_data = real_data.cuda().float()
        
        real_labels = real_data[:, -1].long()
        assert real_labels.min() >= 0 and real_labels.max() < args.num_classes, \
            f"Invalid labels: {real_labels.min()}, {real_labels.max()}. They must be in [0, num_classes-1]"
        img = real_data[:, :-1]  # Lấy tất cả các cột trừ cột nhãn
        lab = real_labels  # Nhãn

        # Áp dụng augmentation ngẫu nhiên (nếu có)
        output = model(img)  # Áp dụng augmentation trước khi đưa vào mô hình
        loss = criterion(output, lab)  # Tính loss giữa đầu ra và nhãn

        optim_model.zero_grad()  # Reset gradient
        loss.backward()  # Tính gradient
        optim_model.step()  # Cập nhật mô hình

        
def train(args, ctgan,modelfolder, modelfilesavepath, generator, discriminator, optim_g, optim_d, trainloader, label_mapping, criterion, aug=None):
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

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        model = define_model(args, args.num_classes).cuda()
        model.train()
        optim_model = op_model(model, args.eval_lr, args.momentum, args.weight_decay)

        for batch_idx, (real_data,) in enumerate(trainloader):
            real_data = real_data.cuda()
            real_data = torch.tensor(real_data, dtype=torch.float32)
            img_real = real_data[:, :-1] 
            real_labels = real_data[:, -1].long()  # Giả sử nhãn là cột cuối
            unique_labels = torch.unique(real_labels)
            # --- Train Generator ---
            optim_g.zero_grad()
            discriminator.eval()  # Ensure discriminator is in evaluation mode

            # Generate synthetic labels and data
            

            #label_counts = dict(pd.Series(gen_labels.cpu().numpy()).value_counts())
            gen_data = ctgan.sample(len(real_data))  # Generate synthetic data
            gen_data = preprocess_data(gen_data)
            gen_data = torch.tensor(gen_data.values, dtype=torch.float32).cuda()  # Convert to tensor and move to GPU
            gen_data = gen_data.cuda() 
            img_syn = gen_data[:, :-1] 
            gen_labels = gen_data[:, -1].long()
            gen_source, gen_class = discriminator.predict(img_syn, args.num_classes) 

            gen_source_loss = -torch.mean(gen_source)                                                                                                            
            gen_class_loss = criterion(gen_class, real_labels)
            gen_loss = gen_source_loss + gen_class_loss

            train_match_model(args, model, optim_model, trainloader, criterion)

            if args.match_aug:
                img_aug = aug(torch.cat([img_real, img_syn]))
                match_loss = matchloss(args, img_aug[:args.batch_size], img_aug[args.batch_size:], real_labels, real_labels, model)
            else:
                match_loss = matchloss(args, img_real, img_syn, real_labels, real_labels, model)
            
            gen_loss = gen_loss + match_loss

            gen_loss.backward()
            optim_g.step()  # Optimize generator
            gen_losses.update(gen_loss.item(), img_real.size(0))

            # --- Train Discriminator ---
            optim_d.zero_grad()
            discriminator.train()

            with torch.no_grad():
                fake_data = img_syn.clone()

            fake_preds, fake_logits = discriminator.predict(fake_data, args.num_classes)
            real_preds, real_logits = discriminator.predict(img_real, args.num_classes)

            acc1 = accuracy(real_logits, real_labels, topk=(1,))[0]
            top1.update(acc1.item(), real_data.size(0))

            fake_source_loss = torch.mean(fake_preds)
            real_source_loss = -torch.mean(real_preds)

            real_class_loss = criterion(real_logits, real_labels)
            fake_class_loss = criterion(fake_logits, gen_labels)

            #gradient_penalty = calc_gradient_penalty(discriminator, real_data, fake_data)
            disc_loss = real_source_loss + fake_source_loss + real_class_loss + fake_class_loss + calc_gradient_penalty(discriminator, img_real,fake_data)

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
    ctgan.save_model(output_dir=modelfolder, file_name = modelfilesavepath)
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


