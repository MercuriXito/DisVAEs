import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.vanilla_vae import VanillaVAE

class FactorVAE(VanillaVAE):
    def __init__(self, 
            in_channels: int, 
            hidden_channels: list, 
            latent_dim: int, 
            input_size: tuple,
            gamma: float,
            **kws):
        
        super(FactorVAE, self).__init__(
            in_channels, hidden_channels, latent_dim, input_size
        )

        self.gamma = gamma
        # Additional Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000), 
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000), 
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000), 
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )

    def get_loss(self, original_x, reconst, mu, logvar, latent,
            *args, **kws):
        """ ELBO of Factor-VAE includes: (i) reconstruction loss; (ii) KL loss; (iii) TC loss 
        estimated by density-ratio trick.
        """
        losses, recorder = super().get_loss(original_x, reconst, mu, logvar, latent, *args, **kws)

        total_loss = losses["total_loss"]

        # density-ratio
        device = latent.device

        latent = latent.detach()
        perm_latent = self.permutate(latent)
        out_original = self.discriminator(latent)
        out_perm = self.discriminator(perm_latent)
        batch_sie = out_original.size(0)
        true_labels = torch.full((batch_sie, ), 1).type(torch.long).to(device)
        fake_labels = torch.zeros((batch_sie, ) ).type(torch.long).to(device)

        tc_loss = (F.cross_entropy(out_original, true_labels) + F.cross_entropy(out_perm, fake_labels)).mean() * 0.5

        total_loss += tc_loss * self.gamma
        losses["tc_loss"] = tc_loss
        losses["total_loss"] = total_loss

        return losses, recorder

    def permutate(self, latent: torch.Tensor):
        batch_size, latent_dim = latent.size()
        perm_latent = torch.zeros_like(latent).to(latent.device)
        for i in range(latent_dim):
            perm = torch.randperm(batch_size)
            perm_latent[:,i] = latent[perm, i]
        return perm_latent
