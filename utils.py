import torch
import os
from scipy import signal

class ResizeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, X, y):
        old_size = X.shape[2]
        X_resized = signal.resample(X, self.new_size, axis=2)
        return X_resized, y

def gradient_penalty(critic, real, alpha, step, fake, device="cpu"):
    BATCH_SIZE, C, H = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, H).to(device)
    interpolated_images = real * beta + fake * (1 - beta)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, step,)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    # one-sided penalty
    gradient_penalty = torch.mean((torch.max(gradient_norm - 1, torch.zeros_like(gradient_norm))) ** 2)
    return gradient_penalty

def save_checkpoint(step, epoch, model, optimizer, filename):
    """
    usage:
        save_checkpoint(epoch, gen, opt_gen, "generator_checkpoint.pth")
        save_checkpoint(epoch, critic, opt_critic, "critic_checkpoint.pth")
    """
    filename = filename.replace('.pth', f'_{step}_{epoch}.pth')
    path = os.path.join("models", filename)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    print('=> Saving checkpoint')
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, filename):
    """
    usage:
        start_epoch = load_checkpoint(gen, opt_gen, "generator_checkpoint.pth")
        start_epoch = max(start_epoch, load_checkpoint(critic, opt_critic, "critic_checkpoint.pth"))

        for epoch in range(start_epoch, NUM_EPOCHS):
            # ...
            # your training code here
            # ...
    """
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]  # You can use this to know where the training left off