import torch
from torch import nn

class PixelNorm1D(nn.Module):
    def __init__(self):
        super(PixelNorm1D, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm1D()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor)  # Upsample
        x = self.lrelu(self.pixel_norm(self.conv1(x)))
        x = self.lrelu(self.pixel_norm(self.conv2(x)))
        return x

class EEG_GAN_Generator(nn.Module):
    def __init__(self, latent_dim):
        super(EEG_GAN_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 50*7) # Adjusted for 401 input
        self.main = nn.Sequential(
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50, scale_factor=1.791), # 1.79017857 1.4375
            nn.Conv1d(50, 3, kernel_size=1) # Changed the input channels to 3
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim)
        z = self.linear(z)
        z = z.view(-1, 50, 7) # Adjusted for 401
        return self.main(z)
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.avg_pool(x)
        return x

class EEG_GAN_Discriminator(nn.Module):
    def __init__(self):
        super(EEG_GAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(3, 50, kernel_size=1), # Changed the input channels to 3
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            nn.Flatten(),
            #nn.Linear(50*13, 1) # Adjusted for 401 input
            nn.Linear(50*12, 1)
        )

    def forward(self, x):
        return self.main(x)




















class ProgressiveEEG_GAN_Generator(nn.Module):
    def __init__(self, latent_dim):
        super(ProgressiveEEG_GAN_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 50*7)
        self.blocks = nn.ModuleList([
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50),
            UpsampleBlock(50, 50, scale_factor=1.791),
        ])
        self.to_rgb = nn.Conv1d(50, 3, kernel_size=1)

    def forward(self, z, depth):
        z = z.view(-1, self.latent_dim)
        z = self.linear(z)
        z = z.view(-1, 50, 7)
        for i in range(depth + 1):
            z = self.blocks[i](z)
        return self.to_rgb(z)

class ProgressiveEEG_GAN_Discriminator(nn.Module):
    def __init__(self):
        super(ProgressiveEEG_GAN_Discriminator, self).__init__()
        self.blocks = nn.ModuleList([
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
            DownsampleBlock(50, 50),
        ])
        self.from_rgb = nn.Conv1d(3, 50, kernel_size=1)
        self.linear = nn.Linear(50*12, 1)

    def forward(self, x, depth):
        x = self.from_rgb(x)
        for i in range(depth, -1, -1):
            x = self.blocks[i](x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
