import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
#from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import EEG_GAN_Discriminator, EEG_GAN_Generator
from utils import gradient_penalty

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
CHANNELS_SIGNALS = 1 # 3 if cz, c4 and c3
Z_DIM = 200
NUM_EPOCHS = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Load your EEG data instead of image data
dataset = MyEEGDataset(...)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = EEG_GAN_Generator(Z_DIM).to(device)
critic = EEG_GAN_Discriminator().to(device)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

fixed_noise = torch.randn(32, Z_DIM, 1).to(device)
writer_real = SummaryWriter(f"logs/EEG/real")
writer_fake = SummaryWriter(f"logs/EEG/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=device)
            w_dist = torch.mean(critic_real) - torch.mean(critic_fake)
            # Update the critic loss function
            loss_critic = -w_dist + torch.max(w_dist, torch.zeros_like(w_dist)) * LAMBDA_GP * gp
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)

                if CHANNELS_SIGNALS > 1:
                    fig, axs = plt.subplots(CHANNELS_SIGNALS*2)
                    fig.suptitle('Real and Fake EEG signals')
                    for i in range(CHANNELS_SIGNALS):
                        axs[i].plot(real[0, i, :].cpu().numpy())
                        axs[i].set_title(f"Real Channel {i+1}")
                        axs[i+CHANNELS_SIGNALS].plot(fake[0, i, :].cpu().numpy())
                        axs[i+CHANNELS_SIGNALS].set_title(f"Fake Channel {i+1}")
                else:
                    fig, axs = plt.subplots(2)
                    fig.suptitle('Real and Fake EEG signals')
                    axs[0].plot(real[0].cpu().numpy())
                    axs[0].set_title("Real")
                    axs[1].plot(fake[0].cpu().numpy())
                    axs[1].set_title("Fake")

                writer_real.add_figure("EEG signals", fig, global_step=step)

            step += 1