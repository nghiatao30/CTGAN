"""CTGAN module."""

import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from torch.utils.data import DataLoader, TensorDataset


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def predict(self, input_, num_classes):
        """Returns gen_source and gen_class with correct size based on num_classes."""
        if isinstance(input_, TensorDataset):
            input_ = input_.tensors[0]

        if not isinstance(input_, torch.Tensor):
            input_ = torch.tensor(input_, dtype=torch.float32)
        input_ = input_.to(self.seq[0].weight.device)

        logits = self.seq(input_)  # Truyền qua mạng để nhận logits
        logits = logits.to(self.seq[0].weight.device)
        if num_classes == 1:
            # Binary classification (real/fake)
            gen_source = torch.sigmoid(logits)  # Lấy giá trị đầu tiên cho real/fake
            gen_class = torch.sigmoid(logits)  # gen_class có kích thước [batch_size, 2]
        else:
            # Multi-class classification
            gen_source = Linear(logits.size(1), 1).to(self.seq[0].weight.device)(logits)  # Không có gen_source trong multi-class
            # Tạo gen_class với kích thước [batch_size, num_classes]
            gen_class = Linear(logits.size(1), num_classes).to(self.seq[0].weight.device)(logits)  # Lấy logits cho các lớp từ thứ hai trở đi

        return gen_source, gen_class

    
    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))  # Truyền x qua chuỗi các lớp
    
class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=200,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=20,
        pac=10,
        cuda=True,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self.optim_G = None
        self.optim_D = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
        
    # def getDiscriminator(self, subpac):
    #     data_dim = self._transformer.output_dimensions
    #     print("data_dim shape: ", data_dim)
    #     print("data_discri shape: ", (data_dim + self._data_sampler.dim_cond_vec()))
    #     return Discriminator(
    #        data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=subpac
    #     ).to(self._device)
    
    def getDiscriminator(self, sample_data, subpac):
        # Chuyển đổi sample_data thành tensor nếu nó là numpy array
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.tensor(sample_data)

        # Xác định input_dim dựa trên kích thước của sample_data
        input_dim = sample_data.shape[1] if sample_data.ndim > 1 else sample_data.size(0)
        input_dim *= subpac  # Điều chỉnh input_dim theo pac nếu cần thiết

        print("input_dim shape: ", input_dim)

        # Khởi tạo và trả về Discriminator với input_dim đã điều chỉnh
        return Discriminator(
            input_dim=input_dim, discriminator_dim=self._discriminator_dim, pac=subpac
        ).to(self._device)

    # def getDiscriminator(self, sample_data, subpac):
    #     # Chuyển đổi sample_data thành tensor nếu nó là numpy array
    #     if isinstance(sample_data, np.ndarray):
    #         sample_data = torch.tensor(sample_data)

    #     # Xác định input_dim dựa trên kích thước của sample_data
    #     input_dim = sample_data.shape[1] if sample_data.ndim > 1 else sample_data.size(0)
    #     input_dim *= subpac  # Điều chỉnh input_dim theo pac nếu cần thiết

    #     print("input_dim shape: ", input_dim)

    #     # Gọi self._discriminator thay vì tạo mới
    #     # Điều chỉnh lại input_dim và pac dựa trên tham số
    #     self._discriminator.input_dim = input_dim  # Điều chỉnh lại input_dim cho _discriminator
    #     self._discriminator.pac = subpac  # Điều chỉnh lại pac cho _discriminator

    #     # Trả về discriminator đã được điều chỉnh
    #     return self._discriminator.to(self._device)

        
    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        # Initialize generator and discriminator
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        # Initialize optimizers for generator and discriminator
        self.optim_G = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        self.optim_D = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.optim_D.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optim_D.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 6
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optim_G.zero_grad(set_to_none=False)
                loss_g.backward()
                self.optim_G.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
        file_name = f'ctgan_model_epoch_{self._epochs}.pth'
        self.save_model(output_dir="2018", file_name=file_name)


    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    @random_state
    def sample_by_label(self, n, label_column, label_value):
        """
        Sample data from a specific label in the dataset.

        Args:
            n (int): Number of rows to sample.
            label_column (string): Name of the column representing the labels.
            label_value (string or int): Specific label value to condition the sampling on.

        Returns:
            pandas.DataFrame: Sampled data matching the specified label.
        """
        if not self._transformer:
            raise ValueError("The model must be trained before sampling.")

        # Lấy thông tin nhãn từ transformer
        condition_info = self._transformer.convert_column_name_value_to_id(
            label_column, label_value
        )
        global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
            condition_info, self._batch_size
        )

        steps = (n + self._batch_size - 1) // self._batch_size  # Đảm bảo đủ số lượng mẫu
        data = []

        total_samples = 0  # Theo dõi số lượng mẫu đã sinh

        for _ in range(steps):
            # Sinh noise
            mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            # Áp dụng điều kiện nhãn
            condvec = global_condition_vec.copy()
            condvec = torch.from_numpy(condvec).to(self._device)
            fakez = torch.cat([fakez, condvec], dim=1)

            # Sinh dữ liệu
            with torch.no_grad():
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                sampled_data = fakeact.detach().cpu().numpy()

            data.append(sampled_data)
            total_samples += len(sampled_data)

            # Dừng nếu đủ số lượng mẫu
            if total_samples >= n:
                break

        # Gộp dữ liệu và lấy đúng số lượng mẫu
        data = np.concatenate(data, axis=0)[:n]

        # Chuyển dữ liệu về định dạng ban đầu
        return self._transformer.inverse_transform(data)


    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def save_model(self, output_dir="output", file_name = "model_saved"):
        """Lưu mô hình CTGAN vào tệp .pth sau khi huấn luyện."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join(output_dir, file_name)
        torch.save({
            'generator_state_dict': self._generator.state_dict(),
            'discriminator_state_dict': self._discriminator.state_dict(),  # Lưu trạng thái của discriminator
            'optim_g_state_dict': self.optim_G.state_dict(),  # Lưu trạng thái của optimizer generator
            'optim_d_state_dict': self.optim_D.state_dict(),  # Lưu trạng thái của optimizer discriminator
            'transformer': self._transformer,
            'data_sampler': self._data_sampler,
            'config': {
            'embedding_dim': self._embedding_dim,
            'generator_dim': self._generator_dim,
            'discriminator_dim': self._discriminator_dim,
            'batch_size': self._batch_size,
            'pac': self.pac,
            'device': str(self._device)
            }
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path, device='cuda'):
        """Tải mô hình CTGAN từ tệp .pth."""
        checkpoint = torch.load(model_path, map_location=device)
        
        # Khôi phục cấu hình từ checkpoint
        self._embedding_dim = checkpoint['config']['embedding_dim']
        self._generator_dim = checkpoint['config']['generator_dim']
        self._discriminator_dim = checkpoint['config']['discriminator_dim']
        self._batch_size = checkpoint['config']['batch_size']
        self.pac = checkpoint['config']['pac']
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo các thành phần mô hình
        data_dim = checkpoint['transformer'].output_dimensions
        self._transformer = checkpoint['transformer']
        self._data_sampler = checkpoint['data_sampler']
        
        # Khởi tạo generator và discriminator với cấu hình đã lưu
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), 
            self._generator_dim, 
            data_dim
        ).to(self._device)
        self._generator.load_state_dict(checkpoint['generator_state_dict'])
        
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), 
            self._discriminator_dim, 
            pac=self.pac
        ).to(self._device)
        
        print(f"Model loaded from {model_path}")

def sample_by_percentage(original_data, label_column, percentage, generator, label_column_name):

    grouped = original_data.groupby(label_column)
    sampled_data = []
    
    for label, group in grouped:
        # Determine the number of samples to generate
        sample_size = int(len(group) * percentage / 100)
        if sample_size > 0:
            # Generate synthetic samples using the CTGAN model
            synthetic_samples = generator.sample_by_label(
                n=sample_size,
                label_column=label_column_name,
                label_value=label
            )
            sampled_data.append(synthetic_samples)

    # Concatenate the sampled data and return as a single DataFrame
    sampled_data = pd.concat(sampled_data, ignore_index=True)
    return sampled_data
