import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio import build_window, build_optimal_window

EPS = 1e-12

class FourierEncoder(nn.Module):
    def __init__(self, n_basis, kernel_size, stride=None, window_fn='hann', trainable=False, trainable_phase=False, onesided=True, return_complex=True):
        super().__init__()

        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.trainable, self.trainable_phase = trainable, trainable_phase
        self.onesided, self.return_complex = onesided, return_complex

        omega = 2 * math.pi * torch.arange(n_basis // 2 + 1) / n_basis
        n = torch.arange(kernel_size)

        window = build_window(kernel_size, window_fn=window_fn)

        self.frequency, self.time_seq = nn.Parameter(omega, requires_grad=trainable), nn.Parameter(n, requires_grad=False)
        self.window = nn.Parameter(window)

        if self.trainable_phase:
            phi = torch.zeros(n_basis // 2 + 1)
            self.phase = nn.Parameter(phi, requires_grad=True)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, 1, T)
        Returns:
            output <torch.Tensor>:
                Complex tensor with shape of (batch_size, n_basis // 2 + 1, n_frames) if onesided=True and return_complex=True
                Complex tensor with shape of (batch_size, n_basis, n_frames) if onesided=False and return_complex=True
                Tensor with shape of (batch_size, 2 * (n_basis // 2 + 1), n_frames) if onesided=True and return_complex=False
                Tensor with shape of (batch_size, 2 * n_basis, n_frames) if onesided=False and return_complex=False
        """
        n_basis = self.n_basis
        stride = self.stride
        omega, n = self.frequency, self.time_seq
        window = self.window

        omega_n = omega.unsqueeze(dim=1) * n.unsqueeze(dim=0)
        if self.trainable_phase:
            phi = self.phase
            basis_real, basis_imag = torch.cos(-(omega_n + phi.unsqueeze(dim=1))), torch.sin(-(omega_n + phi.unsqueeze(dim=1)))
        else:
            basis_real, basis_imag = torch.cos(-omega_n), torch.sin(-omega_n)
        basis_real, basis_imag = basis_real.unsqueeze(dim=1), basis_imag.unsqueeze(dim=1)

        if not self.onesided:
            _, basis_real_conj, _ = torch.split(basis_real, [1, n_basis // 2 - 1, 1], dim=0)
            _, basis_imag_conj, _ = torch.split(basis_imag, [1, n_basis // 2 - 1, 1], dim=0)
            basis_real_conj, basis_imag_conj = torch.flip(basis_real_conj, dims=(0,)), torch.flip(basis_imag_conj, dims=(0,))
            basis_real, basis_imag = torch.cat([basis_real, basis_real_conj], dim=0), torch.cat([basis_imag, - basis_imag_conj], dim=0)
        basis_real, basis_imag = window * basis_real, window * basis_imag
        output_real, output_imag = F.conv1d(input, basis_real, stride=stride), F.conv1d(input, basis_imag, stride=stride)
        output = torch.cat([output_real.unsqueeze(dim=-1), output_imag.unsqueeze(dim=-1)], dim=-1)

        if self.return_complex:
            output = torch.view_as_complex(output)
        else:
            batch_size, n_bins, n_frames, _ = output.size()
            output = output.permute(0, 3, 1, 2).contiguous()
            output = output.view(batch_size, 2*n_bins, n_frames)

        return output

    def extra_repr(self):
        s = "{n_basis}, kernel_size={kernel_size}, stride={stride}, trainable={trainable}, onesided={onesided}, return_complex={return_complex}"
        if self.trainable_phase:
            s += ", trainable_phase={trainable_phase}"

        return s.format(**self.__dict__)

    def get_basis(self):
        n_basis = self.n_basis
        omega, n = self.frequency, self.time_seq
        window = self.window

        omega_n = omega.unsqueeze(dim=1) * n.unsqueeze(dim=0)
        if self.trainable_phase:
            phi = self.phase
            basis_real, basis_imag = torch.cos(-(omega_n + phi.unsqueeze(dim=1))), torch.sin(-(omega_n + phi.unsqueeze(dim=1)))
        else:
            basis_real, basis_imag = torch.cos(-omega_n), torch.sin(-omega_n)

        if not self.onesided:
            _, basis_real_conj, _ = torch.split(basis_real, [1, n_basis // 2 - 1, 1], dim=0)
            _, basis_imag_conj, _ = torch.split(basis_imag, [1, n_basis // 2 - 1, 1], dim=0)
            basis_real_conj, basis_imag_conj = torch.flip(basis_real_conj, dims=(0,)), torch.flip(basis_imag_conj, dims=(0,))
            basis_real, basis_imag = torch.cat([basis_real, basis_real_conj], dim=0), torch.cat([basis_imag, - basis_imag_conj], dim=0)

        basis_real, basis_imag = window * basis_real, window * basis_imag
        basis = torch.cat([basis_real, basis_imag], dim=0)

        return basis

class FourierDecoder(nn.Module):
    def __init__(self, n_basis, kernel_size, stride=None, window_fn='hann', trainable=False, trainable_phase=False, onesided=True):
        super().__init__()

        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.trainable, self.trainable_phase = trainable, trainable_phase
        self.onesided = onesided

        omega = 2 * math.pi * torch.arange(n_basis // 2 + 1) / n_basis
        n = torch.arange(kernel_size)

        window = build_window(kernel_size, window_fn=window_fn)
        optimal_window = build_optimal_window(window, hop_length=stride)

        self.frequency, self.time_seq = nn.Parameter(omega, requires_grad=trainable), nn.Parameter(n, requires_grad=False)
        self.optimal_window = nn.Parameter(optimal_window)

        if self.trainable_phase:
            phi = torch.zeros(n_basis // 2 + 1)
            self.phase = nn.Parameter(phi, requires_grad=True)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>:
                Complex tensor with shape of (batch_size, kernel_size // 2 + 1, n_frames) if onesided=True
                Complex tensor with shape of (batch_size, kernel_size, n_frames) if onesided=False
                Tensor with shape of (batch_size, 2 * (kernel_size // 2 + 1), n_frames) if onesided=True
                Tensor with shape of (batch_size, 2 * kernel_size, n_frames) if onesided=False
        Returns:
            output <torch.Tensor>: (batch_size, 1, T)
        """
        n_basis = self.n_basis
        stride = self.stride
        omega, n = self.frequency, self.time_seq
        optimal_window = self.optimal_window

        if torch.is_complex(input):
            input_real, input_imag = input.real, input.imag
        else:
            n_bins = input.size(1)
            input_real, input_imag = torch.split(input, [n_bins // 2, n_bins // 2], dim=1)

        omega_n = omega.unsqueeze(dim=1) * n.unsqueeze(dim=0)
        if self.trainable_phase:
            phi = self.phase
            basis_real, basis_imag = torch.cos(omega_n + phi.unsqueeze(dim=1)), torch.sin(omega_n + phi.unsqueeze(dim=1))
        else:
            basis_real, basis_imag = torch.cos(omega_n), torch.sin(omega_n)
        basis_real, basis_imag = basis_real.unsqueeze(dim=1), basis_imag.unsqueeze(dim=1)

        _, basis_real_conj, _ = torch.split(basis_real, [1, n_basis // 2 - 1, 1], dim=0)
        _, basis_imag_conj, _ = torch.split(basis_imag, [1, n_basis // 2 - 1, 1], dim=0)
        basis_real_conj, basis_imag_conj = torch.flip(basis_real_conj, dims=(0,)), torch.flip(basis_imag_conj, dims=(0,))
        basis_real, basis_imag = torch.cat([basis_real, basis_real_conj], dim=0), torch.cat([basis_imag, - basis_imag_conj], dim=0)
        basis_real, basis_imag = optimal_window * basis_real, optimal_window * basis_imag
        basis_real, basis_imag = basis_real / n_basis, basis_imag / n_basis

        if self.onesided:
            _, input_real_conj, _ = torch.split(input_real, [1, n_basis // 2 - 1, 1], dim=1)
            _, input_imag_conj, _ = torch.split(input_imag, [1, n_basis // 2 - 1, 1], dim=1)
            input_real_conj, input_imag_conj = torch.flip(input_real_conj, dims=(1,)), torch.flip(input_imag_conj, dims=(1,))
            input_real, input_imag = torch.cat([input_real, input_real_conj], dim=1), torch.cat([input_imag, - input_imag_conj], dim=1)

        output = F.conv_transpose1d(input_real, basis_real, stride=stride) - F.conv_transpose1d(input_imag, basis_imag, stride=stride)

        return output

    def extra_repr(self):
        s = "{n_basis}, kernel_size={kernel_size}, stride={stride}, trainable={trainable}, onesided={onesided}"
        if self.trainable_phase:
            s += ", trainable_phase={trainable_phase}"

        return s.format(**self.__dict__)

    def get_basis(self):
        n_basis = self.n_basis
        omega, n = self.frequency, self.time_seq
        optimal_window = self.optimal_window

        omega_n = omega.unsqueeze(dim=1) * n.unsqueeze(dim=0)
        if self.trainable_phase:
            phi = self.phase
            basis_real, basis_imag = torch.cos(omega_n + phi.unsqueeze(dim=1)), torch.sin(omega_n + phi.unsqueeze(dim=1))
        else:
            basis_real, basis_imag = torch.cos(omega_n), torch.sin(omega_n)

        if not self.onesided:
            _, basis_real_conj, _ = torch.split(basis_real, [1, n_basis // 2 - 1, 1], dim=0)
            _, basis_imag_conj, _ = torch.split(basis_imag, [1, n_basis // 2 - 1, 1], dim=0)
            basis_real_conj, basis_imag_conj = torch.flip(basis_real_conj, dims=(0,)), torch.flip(basis_imag_conj, dims=(0,))
            basis_real, basis_imag = torch.cat([basis_real, basis_real_conj], dim=0), torch.cat([basis_imag, - basis_imag_conj], dim=0)

        basis_real, basis_imag = optimal_window * basis_real, optimal_window * basis_imag
        basis_real, basis_imag = basis_real / n_basis, basis_imag / n_basis
        basis = torch.cat([basis_real, basis_imag], dim=0)

        return basis

class Encoder(nn.Module):
    def __init__(self, in_channels, n_basis, kernel_size=16, stride=8, nonlinear=None):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride
        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(in_channels, n_basis, kernel_size=kernel_size, stride=stride, bias=False)
        if nonlinear is not None:
            if nonlinear == 'relu':
                self.nonlinear1d = nn.ReLU()
            else:
                raise NotImplementedError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

    def forward(self, input):
        x = self.conv1d(input)

        if self.nonlinear:
            output = self.nonlinear1d(x)
        else:
            output = x

        return output

    def get_basis(self):
        basis = self.conv1d.weight

        return basis

class Decoder(nn.Module):
    def __init__(self, n_basis, out_channels, kernel_size=16, stride=8):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride

        self.conv_transpose1d = nn.ConvTranspose1d(n_basis, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, input):
        output = self.conv_transpose1d(input)
        return output

    def get_basis(self):
        basis = self.conv_transpose1d.weight
        return basis

class PinvDecoder(nn.Module):
    def __init__(self, encoder: Union[Encoder, FourierEncoder]):
        super().__init__()

        self.encoder = encoder
        self.kernel_size, self.stride = encoder.kernel_size, encoder.stride

        if isinstance(encoder, Encoder):
            if encoder.nonlinear:
                raise ValueError("Not support pseudo inverse of 'Conv1d + nonlinear'.")
            self.weight = encoder.conv1d.weight

            n_rows, _, n_columns = self.weight.size()
            if n_rows < n_columns:
                raise ValueError("Cannot compute the left inverse of encoder's weight. In encoder, `out_channels` must be equal to or greater than `kernel_size`.")
        elif isinstance(encoder, FourierEncoder):
            if encoder.onesided or encoder.return_complex:
                raise ValueError("Both encoder.onesided and encoder.return_complex are expected to be False.")
        else:
            raise TypeError("Invalid encoder is given.")

    def forward(self, input):
        encoder = self.encoder
        kernel_size, stride = self.kernel_size, self.stride
        duplicate = kernel_size // stride

        if isinstance(encoder, Encoder):
            weight = self.weight.permute(1, 0, 2).contiguous()
            weight_pinverse = torch.pinverse(weight).permute(2, 0, 1).contiguous() / duplicate
            output = F.conv_transpose1d(input, weight_pinverse, stride=stride)
        elif isinstance(encoder, FourierEncoder):
            if torch.is_complex(input):
                input_real, input_imag = input.real, input.imag
            else:
                n_bins = input.size(1)
                input_real, input_imag = torch.split(input, [n_bins // 2, n_bins // 2], dim=1)

            n_basis = encoder.n_basis
            omega, n = encoder.frequency, encoder.time_seq
            window = encoder.window

            omega_n = omega.unsqueeze(dim=1) * n.unsqueeze(dim=0)

            if encoder.trainable_phase:
                phi = self.encoder.phase
                basis_real, basis_imag = torch.cos(omega_n + phi.unsqueeze(dim=1)), torch.sin(omega_n + phi.unsqueeze(dim=1))
            else:
                basis_real, basis_imag = torch.cos(omega_n), torch.sin(omega_n)
            basis_real, basis_imag = basis_real.unsqueeze(dim=1), basis_imag.unsqueeze(dim=1)

            _, basis_real_conj, _ = torch.split(basis_real, [1, n_basis // 2 - 1, 1], dim=0)
            _, basis_imag_conj, _ = torch.split(basis_imag, [1, n_basis // 2 - 1, 1], dim=0)
            basis_real_conj, basis_imag_conj = torch.flip(basis_real_conj, dims=(0,)), torch.flip(basis_imag_conj, dims=(0,))
            basis_real, basis_imag = torch.cat([basis_real, basis_real_conj], dim=0), torch.cat([basis_imag, - basis_imag_conj], dim=0)
            basis_real, basis_imag = window * basis_real, window * basis_imag
            basis_real, basis_imag = basis_real / n_basis, basis_imag / n_basis
            output = F.conv_transpose1d(input_real, basis_real, stride=stride) - F.conv_transpose1d(input_imag, basis_imag, stride=stride)
        else:
            raise TypeError("Not support encoder {}.".format(type(encoder)))

        return output

    def get_basis(self):
        kernel_size, stride = self.kernel_size, self.stride
        duplicate = kernel_size // stride
        weight = self.weight.permute(1, 0, 2).contiguous()
        weight_pinverse = torch.pinverse(weight).permute(2, 0, 1).contiguous() / duplicate

        basis = weight_pinverse

        return basis

class GatedEncoder(nn.Module):
    def __init__(self, in_channels, n_basis, kernel_size=16, stride=8, eps=EPS):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride
        self.eps = eps

        self.conv1d_U = nn.Conv1d(in_channels, n_basis, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1d_V = nn.Conv1d(in_channels, n_basis, kernel_size=kernel_size, stride=stride, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        eps = self.eps

        norm = torch.linalg.norm(input, dim=2, keepdim=True)
        x = input / (norm + eps)
        x_U = self.conv1d_U(x)
        x_V = self.conv1d_V(x)
        output = self.relu(x_U) * self.sigmoid(x_V)

        return output

def _test_filterbank():
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    n_basis = kernel_size

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    print("-"*10, "Trainable Encoder", "-"*10)
    encoder = Encoder(C, 2*kernel_size, kernel_size=kernel_size, stride=stride)
    decoder = Decoder(2*kernel_size, C, kernel_size=kernel_size, stride=stride)

    enc_basis, dec_basis = encoder.get_basis(), decoder.get_basis()

    plt.figure()
    plt.pcolormesh(enc_basis.squeeze(dim=1).detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/filterbank/basis_enc-trainable.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.pcolormesh(dec_basis.squeeze(dim=1).detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/filterbank/basis_dec-trainable.png', bbox_inches='tight')
    plt.close()

    latent = encoder(input)
    output = decoder(latent)
    print(input.size(), output.size(), latent.size())
    print()

    print("-"*10, "Fourier Encoder (onesided=True, return_complex=True)", "-"*10)
    n_basis = kernel_size
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, onesided=True, return_complex=True)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, onesided=True)

    enc_basis, dec_basis = encoder.get_basis(), decoder.get_basis()

    plt.figure()
    plt.pcolormesh(enc_basis.squeeze(dim=1).detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/filterbank/basis_enc-Fourier.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.pcolormesh(dec_basis.squeeze(dim=1).detach().cpu().numpy(), cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar()
    plt.savefig('data/filterbank/basis_dec-Fourier.png', bbox_inches='tight')
    plt.close()

    spectrogram = encoder(input)
    output = decoder(spectrogram)
    print(input.size(), output.size(), spectrogram.size())

    plt.figure()
    plt.plot(input[0, 0].detach().numpy())
    plt.plot(output[0, 0].detach().numpy())
    plt.savefig('data/filterbank/Fourier.png', bbox_inches='tight')
    plt.close()

    print("-"*10, "Fourier Encoder (onesided=False, return_complex=True)", "-"*10)
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, onesided=False, return_complex=True)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, onesided=False)

    spectrogram = encoder(input)
    output = decoder(spectrogram)
    print(input.size(), output.size(), spectrogram.size())
    print()

    print("-"*10, "Fourier Encoder (onesided=False, return_complex=False)", "-"*10)
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, onesided=False, return_complex=False)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, onesided=False)

    spectrogram = encoder(input)
    output = decoder(spectrogram)
    print(input.size(), output.size(), spectrogram.size())
    print()

    print("-"*10, "Encoder and pseudo inverse of encoder", "-"*10)
    encoder = Encoder(C, n_basis, kernel_size, stride=stride)
    decoder = PinvEncoder(encoder)
    latent = encoder(input)
    output = decoder(latent)
    print(input.size(), output.size())

    plt.figure()
    plt.plot(range(T), input[0, 0].detach().numpy())
    plt.plot(range(T), output[0, 0].detach().numpy())
    plt.savefig('data/filterbank/pinv.png', bbox_inches='tight')
    plt.close()

def _test_fourier():
    batch_size = 2
    C = 1
    T = 64
    kernel_size, stride = 8, 2
    onesided, return_complex = True, True

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    print("-"*10, "Fourier Encoder", "-"*10)
    n_basis = kernel_size
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided, return_complex=return_complex)

    spectrogram = encoder(input)
    amplitude = torch.clip(torch.abs(spectrogram), min=EPS)
    magnitude = 20 * torch.log10(amplitude)

    plt.figure()
    plt.pcolormesh(magnitude[0].detach().cpu().numpy(), cmap='bwr')
    plt.colorbar()
    plt.savefig('data/filterbank/spectrogram-enc.png', bbox_inches='tight')
    plt.close()

    window = build_window(kernel_size, window_fn='hann')
    spectrogram_stft = torch.stft(input.view(batch_size * C, T), kernel_size, hop_length=stride, window=window, center=False, onesided=onesided, return_complex=return_complex)
    amplitude_stft = torch.clip(torch.abs(spectrogram_stft), min=EPS)
    magnitude_stft = 20 * torch.log10(amplitude_stft)
    magnitude_stft = magnitude_stft.view(batch_size, C, *magnitude_stft.size()[1:])

    plt.figure()
    plt.pcolormesh(magnitude_stft[0, 0].detach().cpu().numpy(), cmap='bwr')
    plt.colorbar()
    plt.savefig('data/filterbank/spectrogram-stft.png', bbox_inches='tight')
    plt.close()

    n_basis = kernel_size
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided, return_complex=return_complex)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided)
    spectrogram = encoder(input)
    output = decoder(spectrogram)

    plt.figure()
    plt.plot(input[0, 0].detach().numpy())
    plt.plot(output[0, 0].detach().numpy())
    plt.savefig('data/filterbank/reconstruction_N{}_K{}.png'.format(n_basis, kernel_size), bbox_inches='tight')
    plt.close()

    n_basis = kernel_size // 2
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided, return_complex=return_complex)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided)
    spectrogram = encoder(input)
    output = decoder(spectrogram)

    plt.figure()
    plt.plot(input[0, 0].detach().numpy())
    plt.plot(output[0, 0].detach().numpy())
    plt.savefig('data/filterbank/reconstruction_N{}_K{}.png'.format(n_basis, kernel_size), bbox_inches='tight')
    plt.close()

    n_basis = kernel_size * 2
    encoder = FourierEncoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided, return_complex=return_complex)
    decoder = FourierDecoder(n_basis, kernel_size, stride=stride, window_fn='hann', onesided=onesided)
    spectrogram = encoder(input)
    output = decoder(spectrogram)

    plt.figure()
    plt.plot(input[0, 0].detach().numpy())
    plt.plot(output[0, 0].detach().numpy())
    plt.savefig('data/filterbank/reconstruction_N{}_K{}.png'.format(n_basis, kernel_size), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    print("="*10, "Filterbank", "="*10)
    _test_filterbank()
    print()

    print("="*10, "Fourier basis", "="*10)
    _test_fourier()