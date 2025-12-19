"""Autoencoder for MNIST.

Autoencoder adopted from:
https://medium.com/@j.calzaretta.ai/exploring-diffusion-models-a-hands-on-approach-with-mnist-baf79aa4d195

VAE adopted from:
https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

When training the autoencoder currently in `mnist_weights`, the final average
loss was `0.005405`.
"""
import torch
from torch import nn


class AEEncoder(nn.Module):

    def __init__(self):
        """Autoencoder for MNIST."""
        super().__init__()

        channels = [4, 4, 4]

        self.model = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[2]),
        )

    def forward(self, x):
        return self.model(x)


class AEDecoder(nn.Module):

    def __init__(self):
        """Autoencoder for MNIST."""
        super().__init__()

        channels = [4, 4, 4]

        self.model = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1,
                               bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[1],
                               channels[0],
                               3,
                               stride=2,
                               bias=True,
                               output_padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, latent):
        # Reshape (batch_size, 400) to (batch_size, 4, 10, 10).
        if latent.ndim == 2:
            latent = latent.reshape(-1, 4, 10, 10)

        return self.model(latent)


class VAEEncoder(nn.Module):

    def __init__(self):
        """Autoencoder for MNIST."""
        super().__init__()

        channels = [4, 4, 4]

        self.model = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            # Multiply by 2 for mean and variance.
            nn.Conv2d(channels[1], 2 * channels[2], 3, stride=1, bias=True),
            nn.BatchNorm2d(2 * channels[2]),
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.model(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar


class VAEDecoder(nn.Module):

    def __init__(self):
        """Autoencoder for MNIST."""
        super().__init__()

        channels = [4, 4, 4]

        self.model = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1,
                               bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[1],
                               channels[0],
                               3,
                               stride=2,
                               bias=True,
                               output_padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, latent):
        # Reshape (batch_size, 400) to (batch_size, 4, 10, 10).
        if latent.ndim == 2:
            latent = latent.reshape(-1, 4, 10, 10)

        return self.model(latent)


def train(model: str):
    #  from lpips import LPIPS
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from tqdm import tqdm, trange

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "ae":
        encoder = AEEncoder().to(device)
        decoder = AEDecoder().to(device)
    elif model == "vae":
        encoder = VAEEncoder().to(device)
        decoder = VAEDecoder().to(device)
    else:
        raise ValueError("Unknown model")

    n_epochs = 100  #@param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 2048  #@param {'type':'integer'}
    ## learning rate
    lr = 1e-3  #@param {'type':'number'}

    dataset = MNIST(
        '.',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # Moves images to [-1, 1] range instead of [0, 1].
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    total_params = {
        "encoder": sum(p.numel() for p in encoder.parameters()),
        "decoder": sum(p.numel() for p in decoder.parameters()),
    }
    trainable_params = {
        "encoder":
            sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        "decoder":
            sum(p.numel() for p in decoder.parameters() if p.requires_grad),
    }
    nontrainable_params = {
        k: total_params[k] - trainable_params[k] for k in total_params
    }
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {nontrainable_params}")
    print("-----------------------------")

    # Define the loss function, MSE and LPIPS
    # lpips = LPIPS(net="squeeze").cuda()
    def ae_loss(x, xhat):
        return nn.functional.mse_loss(x, xhat)
        # + lpips(x.repeat(1,3,1,1), x_hat.repeat(1,3,1,1)).mean()

    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )
    tqdm_epoch = trange(1, n_epochs + 1)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, _ in dataloader:
            x = x.to(device)
            if model == "ae":
                z = encoder(x)  # pylint: disable = not-callable
                x_hat = decoder(z)  # pylint: disable = not-callable
                loss = ae_loss(x, x_hat)
            elif model == "vae":
                # Encode
                # pylint: disable-next = not-callable
                mu, logvar = encoder(x)

                # Reparameterize
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)

                z = mu + eps * std

                # Decode
                # pylint: disable-next = not-callable
                recon_target = decoder(z)

                # Compute loss
                loss = vae_loss(recon_target, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm.write(f"{epoch} Average Loss: {avg_loss / num_items:5f}")
        # Print the averaged training loss so far.
        tqdm_epoch.set_description(f"Average Loss: {avg_loss / num_items:5f}")

        # Update the checkpoint after each epoch of training.
        #  torch.save(ae_model.state_dict(), f'ckpt_mnist_mse_{n_epochs}e.pth')

    torch.save(encoder.state_dict(), f"{model}_encoder.pth")
    torch.save(decoder.state_dict(), f"{model}_decoder.pth")


if __name__ == "__main__":
    import fire
    fire.Fire(train)
