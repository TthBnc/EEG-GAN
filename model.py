import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

class WSConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=9, stride=1, padding=1, gain=2
    ):
        super(WSConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.scale = (gain / (in_channels * kernel_size)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1)

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        Initialize the PixelNorm Layer.
        :param epsilon: small constant to prevent division by zero
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        """
        Forward pass of the layer.
        :param x: input activations volume
        :return: output activations volume
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv1d(in_channels, out_channels)
        self.conv2 = WSConv1d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class EEG_PRO_GAN_Generator(nn.Module):
    def __init__(self, z_dim, in_channels, initial_signal_size, factors, signal_channels=1):
        super(EEG_PRO_GAN_Generator, self).__init__()

        self.z_dim = z_dim
        self.initial_signal_size = initial_signal_size
        self.in_channels = in_channels
        self.leaky = nn.LeakyReLU(0.2)

        # self.initial = nn.Sequential(
        #     PixelNorm(),
        #     nn.ConvTranspose1d(z_dim, in_channels, 4, 1, 0),
        #     nn.LeakyReLU(0.2),
        #     WSConv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     PixelNorm(),
        # )
        self.initial = nn.Sequential(
            nn.Linear(z_dim, 50*initial_signal_size),
            nn.LeakyReLU(0.2)
        )

        self.initial_signal = WSConv1d(
            in_channels, signal_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.signal_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_signal]),
        )

        for i in range(
            len(factors) - 1 # len(factors) -1 == number of blocks
        ):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.signal_layers.append(
                WSConv1d(conv_out_c, signal_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        """
        Interpolate between the upscaled version of the lower resolution image and the image generated by the current layer.
        :param alpha: alpha value to use for interpolation
        :param upscaled: upscaled version of the lower resolution image
        :param generated: image generated by the current layer
        :return: interpolated image
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)


    def forward(self, x, alpha, steps, scale_factors):
        # during training before calling gen
        # need to check if we are at the last step or not
        x = x.view(-1, self.z_dim)
        out = self.leaky(self.initial(x))
        out = out.view(-1, 50, self.initial_signal_size)

        if steps == 0:
            return self.initial_signal(out)

        for step in range(steps):
            scale_factor = scale_factors[step]

            upscaled = F.interpolate(out, scale_factor=scale_factor, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.signal_layers[steps - 1](upscaled)
        final_out = self.signal_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)



class EEG_PRO_GAN_Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, initial_signal_size, factors, signal_channels=1):
        super(EEG_PRO_GAN_Discriminator, self).__init__()

        self.initial_signal_size = initial_signal_size
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
            WSConv1d(in_channels + 1, in_channels, kernel_size=9),
            nn.LeakyReLU(0.2),
            WSConv1d(in_channels, in_channels, kernel_size=9),
            nn.LeakyReLU(0.2),
           
            # ConvBlock(in_channels, in_channels, use_pixelnorm=False),

            # WSConv1d(
            #     in_channels, 1, kernel_size=1, padding=0, stride=1
            # ),

            nn.Flatten(),
            # # nn.Linear(in_channels*12, 1),
            nn.Linear(in_channels*initial_signal_size, 1),
        )

    def minibatch_std(self, x):
        """
        Add standard deviation of the minibatch as a feature map.
        :param x: input activations volume
        :return: output activations volume with minibatch stddev feature map
        """
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2])
        return torch.cat([x, batch_statistics], dim=1)

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * out + (1 - alpha) * downscaled)
    
    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.signal_layers[cur_step](x))

        if steps == 0:
            # print(f'1 {out.shape}')
            out = self.minibatch_std(out)
            # print(f'2 {out.shape}')
            out = self.final_block(out)
            # print(f'3 {out.shape}')
            # out = out.view(out.shape[0], -1)
            # print(f'5 {out.shape}')
            return out 

        downscaled = self.leaky(self.signal_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    # signal_sizes = [7, 14, 28, 56, 112, 224, 401]
    real = torch.randn((48, 1, 400)) # from dataloader
    desired_steps = 6
    factors = [1 for _ in range(desired_steps + 1)]
    signal_sizes = []
    scale_factors = []

    for steps in range(desired_steps + 1):
        if steps == 0:
            signal_sizes.append(real.shape[2])
        else:
            signal_sizes.append(signal_sizes[steps-1] // 2)

    signal_sizes.sort()

    last_signal_size = 0
    for signal_size in signal_sizes:
        if not signal_size == signal_sizes[0]:
            scale_factors.append(signal_size / last_signal_size)
        last_signal_size = signal_size

    Z_DIM = 200
    IN_CHANNELS = 50
    gen = EEG_PRO_GAN_Generator(Z_DIM, IN_CHANNELS, signal_sizes[0], signal_channels=1)
    critic = EEG_PRO_GAN_Discriminator(Z_DIM, IN_CHANNELS, signal_sizes[0], signal_channels=1)

    for idx, signal_size in enumerate(signal_sizes):
        # num_steps = int(log2(signal_size / 4))
        num_steps = idx
        x = torch.randn((1, Z_DIM, 1))
        # if signal_size == signal_sizes - 1: # final block
        #     z = gen(x, 0.5, steps=num_steps, final_step=True)
        # else:
        #     z = gen(x, 0.5, steps=num_steps)
        z = gen(x, 0.5, steps=num_steps, scale_factors=scale_factors)
        assert z.shape == (1, 1, signal_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At signal size: {signal_size}")