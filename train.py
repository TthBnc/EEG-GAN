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

from model import EEG_PRO_GAN_Generator, EEG_PRO_GAN_Discriminator
from src.dataset import MI_Dataset_ALL
from utils import ResizeTransform, gradient_penalty, save_checkpoint, load_checkpoint

def train_fn(critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
            tensorboard_step,
            writer,
            scaler_gen,
            scaler_critic,
            epoch,
            num_epochs,
            fixed_noise,
            DEVICE,
            CRITIC_ITERATIONS,
            Z_DIM,
            SCALE_FACTORS,
            LAMBDA_GP,
            PROGRESSIVE_EPOCHS,
            SIGNAL_CHANNELS):
    loop = tqdm(loader)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        with torch.cuda.amp.autocast():
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1).to(DEVICE)
                fake = gen(noise, alpha=alpha, steps=step, scale_factors=SCALE_FACTORS)
                critic_real = critic(real, alpha=alpha, steps=step).reshape(-1)
                # critic_fake = critic(fake.detach(), alpha=alpha, steps=step).reshape(-1)
                critic_fake = critic(fake, alpha=alpha, steps=step).reshape(-1)

                gp = gradient_penalty(critic, real, alpha, step, fake, device=DEVICE)
                w_dist = torch.mean(critic_real) - torch.mean(critic_fake)
                # Update the critic loss function
                loss_critic = -w_dist + torch.max(w_dist, torch.zeros_like(w_dist)) * LAMBDA_GP * gp
            
        # opt_critic.zero_grad()
        # scaler_critic.scale(loss_critic).backward()
        # scaler_critic.step(opt_critic)
        # scaler_critic.update()
        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha=alpha, steps=step).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

        # opt_gen.zero_grad()
        # scaler_gen.scale(loss_gen).backward()
        # scaler_gen.step(opt_gen)
        # scaler_gen.update()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        # Print losses occasionally and print to tensorboard
        if batch_idx % 21 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise, alpha=alpha, steps=step, scale_factors=SCALE_FACTORS)

                if SIGNAL_CHANNELS > 1:
                    fig, axs = plt.subplots(SIGNAL_CHANNELS*2)
                    fig.suptitle('Real and Fake EEG signals')
                    for i in range(SIGNAL_CHANNELS):
                        axs[i].plot(real[0, i, :].cpu().numpy())
                        axs[i].set_title(f"Real Channel {i+1}")
                        axs[i+SIGNAL_CHANNELS].plot(fake[0, i, :].cpu().numpy())
                        axs[i+SIGNAL_CHANNELS].set_title(f"Fake Channel {i+1}")
                else:
                    fig, axs = plt.subplots(2)
                    fig.suptitle('Real and Fake EEG signals')
                    axs[0].plot(real[0].cpu().numpy())
                    axs[0].set_title("Real")
                    axs[1].plot(fake[0].cpu().numpy())
                    axs[1].set_title("Fake")

                writer.add_figure("EEG signals", fig, global_step=tensorboard_step)

            tensorboard_step += 1
        
        if epoch % 250 == 0:
            save_checkpoint(step, epoch, gen, opt_gen, "ALL_generator_checkpoint.pth")
            save_checkpoint(step, epoch, critic, opt_critic, "ALL_critic_checkpoint.pth")

    return tensorboard_step, alpha

def get_loader(signal_size, batch_size, DATA_FOLDER, DEVICE):
    transform = ResizeTransform(signal_size)
    dataset = MI_Dataset_ALL(DATA_FOLDER, device=DEVICE, verbose=True, transform=transform) # uses all signals by default
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, dataset


def main():
    # Hyperparameters etc.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-2
    # BATCH_SIZE = 48
    SIGNAL_CHANNELS = 22 # 3 if cz, c4 and c3
    IN_CHANNELS = 50
    Z_DIM = 200
    NUM_EPOCHS = 1000 # half for fading half when faded in
    BATCH_SIZES = [96, 96, 96, 96, 80, 64, 48]
    PROGRESSIVE_EPOCHS = [NUM_EPOCHS] * len(BATCH_SIZES)
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    # Dataset parameters
    DATA_FOLDER = "resources/data"
    SUBJECT_IDS = [1] # list from 1-9, MI_DATASET_ALL uses all by default
    SIGNALS = ["feet"] # list of the desired signals (feet, right/left_hand, tounge)

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

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = EEG_PRO_GAN_Generator(Z_DIM, IN_CHANNELS, SIGNAL_SIZES[0], factors=FACTORS, signal_channels=SIGNAL_CHANNELS).to(DEVICE)
    critic = EEG_PRO_GAN_Discriminator(Z_DIM, IN_CHANNELS, SIGNAL_SIZES[0], factors=FACTORS, signal_channels=SIGNAL_CHANNELS).to(DEVICE)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    fixed_noise = torch.randn(BATCH_SIZES[-1], Z_DIM, 1).to(DEVICE)
    writer = SummaryWriter(f"logs/EEG")
    # writer_fake = SummaryWriter(f"logs/EEG/fake")

    gen.train()
    critic.train()

    step = 0 
    tensorboard_step = 0
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(SIGNAL_SIZES[step], BATCH_SIZES[step], DATA_FOLDER, DEVICE)
        print(f"Current image size: {SIGNAL_SIZES[step]}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
                epoch,
                num_epochs,
                fixed_noise,
                DEVICE,
                CRITIC_ITERATIONS,
                Z_DIM,
                SCALE_FACTORS,
                LAMBDA_GP,
                PROGRESSIVE_EPOCHS,
                SIGNAL_CHANNELS
            )

            # if epoch % 250:
            #     save_checkpoint(step, epoch, gen, opt_gen, "generator_checkpoint.pth")
            #     save_checkpoint(step, epoch, critic, opt_critic, "critic_checkpoint.pth")
        save_checkpoint(step, num_epochs, gen, opt_gen, "ALL_generator_checkpoint.pth")
        save_checkpoint(step, num_epochs, critic, opt_critic, "ALL_critic_checkpoint.pth")

        step += 1  # progress to the next img size

if __name__ == '__main__':
    main()


# start_epoch = 0 # if training from scratch
# # start_epoch = load_checkpoint(gen, opt_gen, "generator_checkpoint.pth")
# # start_epoch = max(start_epoch, load_checkpoint(critic, opt_critic, "critic_checkpoint.pth"))

# for epoch in range(start_epoch, NUM_EPOCHS):
#     # Target labels not needed! <3 unsupervised
#     for batch_idx, (real, _) in enumerate(tqdm(loader)):
#         real = real.to(DEVICE)
#         cur_batch_size = real.shape[0]

#         # Train Critic: max E[critic(real)] - E[critic(fake)]
#         # equivalent to minimizing the negative of that
#         for _ in range(CRITIC_ITERATIONS):
#             noise = torch.randn(cur_batch_size, Z_DIM, 1).to(DEVICE)
#             fake = gen(noise)
#             critic_real = critic(real).reshape(-1)
#             critic_fake = critic(fake).reshape(-1)

#             gp = gradient_penalty(critic, real, fake, device=DEVICE)
#             w_dist = torch.mean(critic_real) - torch.mean(critic_fake)
#             # Update the critic loss function
#             loss_critic = -w_dist + torch.max(w_dist, torch.zeros_like(w_dist)) * LAMBDA_GP * gp
            
#             critic.zero_grad()
#             loss_critic.backward(retain_graph=True)
#             opt_critic.step()

#         # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
#         gen_fake = critic(fake).reshape(-1)
#         loss_gen = -torch.mean(gen_fake)
#         gen.zero_grad()
#         loss_gen.backward()
#         opt_gen.step()

#         # Print losses occasionally and print to tensorboard
#         if batch_idx % 10 == 0 and batch_idx > 0:
#             print(
#                 f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
#                   Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
#             )
#             with torch.no_grad():
#                 fake = gen(fixed_noise)

#                 if SIGNAL_CHANNELS > 1:
#                     fig, axs = plt.subplots(SIGNAL_CHANNELS*2)
#                     fig.suptitle('Real and Fake EEG signals')
#                     for i in range(SIGNAL_CHANNELS):
#                         axs[i].plot(real[0, i, :].cpu().numpy())
#                         axs[i].set_title(f"Real Channel {i+1}")
#                         axs[i+SIGNAL_CHANNELS].plot(fake[0, i, :].cpu().numpy())
#                         axs[i+SIGNAL_CHANNELS].set_title(f"Fake Channel {i+1}")
#                 else:
#                     fig, axs = plt.subplots(2)
#                     fig.suptitle('Real and Fake EEG signals')
#                     axs[0].plot(real[0].cpu().numpy())
#                     axs[0].set_title("Real")
#                     axs[1].plot(fake[0].cpu().numpy())
#                     axs[1].set_title("Fake")

#                 writer.add_figure("EEG signals", fig, global_step=tensorboard_step)

#             tensorboard_step += 1
        
#         if epoch % 50:
#             save_checkpoint(epoch, gen, opt_gen, "generator_checkpoint.pth")
#             save_checkpoint(epoch, critic, opt_critic, "critic_checkpoint.pth")

# # for idx, signal_size in enumerate(SIGNAL_SIZES):
# #     num_steps = idx