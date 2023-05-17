import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class WSConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
    ):
        super(WSConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1)

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv1d(in_channels, out_channels)
        self.conv2 = WSConv1d(out_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.lrelu(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class EEG_PRO_GAN_Generator(nn.Module):
    def __init__(self, z_dim, in_channels, signal_channels=3):
        super(EEG_PRO_GAN_Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose1d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_signal = WSConv1d(
            in_channels, signal_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.signal_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_signal]),
        )

        for i in range(6): # number of progressive layers
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])

        previous_prog_blocks = self.prog_blocks
        self.prog_blocks, self.signal_layers = (
        nn.ModuleList([]),
        nn.ModuleList([self.initial_signal]),
        )

        for i in range(
            len(factors) - 1
        ):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            if i<len(previous_prog_blocks):
                self.prog_blocks.append(previous_prog_blocks[i])
            else:
                self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.signal_layers.append(
                WSConv1d(conv_out_c, signal_channels, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_signal(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.signal_layers[steps - 1](upscaled)
        final_out = self.signal_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)



class EEG_GAN_Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, signal_channels=3):
        super(EEG_GAN_Discriminator, self).__init__()

        self.prog_blocks, self.signal_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.signal_layers.append(
                WSConv1d(signal_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        self.initial_signal = WSConv1d(
            signal_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.signal_layers.append(self.initial_signal)
        self.avg_pool = nn.AvgPool1d(
            kernel_size=2, stride=2
        )

        self.final_block = nn.Sequential(
            WSConv1d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv1d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv1d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),
        )

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2])
        return torch.cat([x, batch_statistics], dim=1)

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def forward(self, x, alpha, steps):
        if steps == 0:
            out = self.signal_layers[0](x)
            return self.final_block(out)

        for step in range(steps):
            downscaled = self.signal_layers[steps - step](x)
            x = self.avg_pool(x)
            if step == 0:
                out = downscaled
            out = self.prog_blocks[steps - step - 1](downscaled)

        final_downscaled = self.final_block(self.leaky(out))
        final_out = self.final_block(self.leaky(self.signal_layers[0](x)))
        return self.fade_in(alpha, final_downscaled, final_out)

