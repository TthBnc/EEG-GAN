import torch
from model import EEG_PRO_GAN_Generator
from src.dataset import MI_Dataset_ALL
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


fixed_noise = torch.randn(5, 200, 1).to("cpu")
real = torch.randn((48, 1, 400))
DESIRED_STEPS = 6
FACTORS = [1 for _ in range(DESIRED_STEPS + 1)]
SIGNAL_SIZES = []
SCALE_FACTORS = []

for steps in range(DESIRED_STEPS + 1):
    if steps == 0:
        SIGNAL_SIZES.append(real.shape[2])
    else:
        SIGNAL_SIZES.append(SIGNAL_SIZES[steps-1] // 2)

SIGNAL_SIZES.sort()

last_signal_size = 0
for signal_size in SIGNAL_SIZES:
    if not signal_size == SIGNAL_SIZES[0]:
        SCALE_FACTORS.append(signal_size / last_signal_size)
    last_signal_size = signal_size


dataset = MI_Dataset_ALL("resources/data")
loader = DataLoader(dataset, batch_size=48, shuffle=True)

gen = EEG_PRO_GAN_Generator(200, 50, SIGNAL_SIZES[0], factors=FACTORS, signal_channels=1).to("cpu")
checkpoint = torch.load("models/ALL_generator_checkpoint_6_20.pth", map_location=torch.device('cpu'))
gen.load_state_dict(checkpoint["model_state"])
gen.eval()
fake = gen(fixed_noise, alpha=0.5, steps=6, scale_factors=SCALE_FACTORS)
# fake = torch.squeeze(fake)
for batch_idx, (real, _) in enumerate(loader):
    real = real.to("cpu")
    fig, axs = plt.subplots(8)
    fig.suptitle('Real and Fake EEG signals')
    for i in range(4):
        axs[i].plot(real[0, i, :].cpu().detach().numpy())
        axs[i].set_title(f"Real Channel {i+1}")
        axs[i+4].plot(fake[i, 0, :].cpu().detach().numpy())
        axs[i+4].set_title(f"Fake Channel {i+1}")
    # plt.plot(fake[0, 1, :].cpu().detach().numpy())
    plt.show()
    break
print("done")