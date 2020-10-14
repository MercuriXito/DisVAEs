import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.vanilla_vae import VanillaVAE

class BetaVAE(VanillaVAE):
    def __init__(self, 
            in_channels: int, 
            hidden_channels: list, 
            latent_dim: int, 
            input_size,
            beta: int, **kws):
        super(BetaVAE, self).__init__(
            in_channels, hidden_channels, latent_dim, input_size, **kws
        )
        self.beta = beta

    def get_loss(self, original_x, reconst, mu, logvar, latent,
            *args, **kws):

        losses, recorder = super().get_loss(original_x, reconst, mu, logvar, latent, *args, **kws)

        kl_loss = losses["kl_loss"] * self.beta
        reconst_loss = losses["reconst_loss"]

        losses["kl_loss"] = kl_loss
        losses["total_loss"] = reconst_loss + kl_loss

        recorder["beta"] = self.beta
        return losses, recorder
