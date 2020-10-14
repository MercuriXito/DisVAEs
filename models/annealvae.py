import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.vanilla_vae import VanillaVAE

class AnnealVAE(VanillaVAE):
    def __init__(self, 
            in_channels: int, 
            hidden_channels: list, 
            latent_dim: int, 
            input_size,
            gamma: int,
            Cmax: int,
            interval: int,
            **kws):

        super(AnnealVAE, self).__init__(
            in_channels, hidden_channels, latent_dim, input_size, **kws
        )
        self.gamma = gamma
        self.Cmax = Cmax
        self.interval = interval

    def get_loss(self, original_x, reconst, mu, logvar, latent, num_iter,
            *args, **kws):

        # increse capacity in every interval iterations
        capacity = min(num_iter // self.interval, self.Cmax)

        # loss from vanilla_vae 
        losses, recorder = super().get_loss(original_x, reconst, mu, logvar, latent, *args, **kws)

        kl_loss = losses["kl_loss"]
        reconst_loss = losses["reconst_loss"]
        kl_loss = torch.abs(kl_loss - capacity) * self.gamma

        losses["kl_loss"] = kl_loss
        losses["total_loss"] = reconst_loss + kl_loss

        recorder["capacity"] = capacity
        return losses, recorder
