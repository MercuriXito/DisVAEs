import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.vanilla_vae import VanillaVAE

class DipVAE(VanillaVAE):
    def __init__(self, 
        in_channels: int, 
        hidden_channels: list, 
        latent_dim: int, 
        input_size: tuple,
        lambda_od: float, 
        lambda_d: float,
        use_full_cov = False,
        **kws):
        """
        DIP-VAE model, set `use_full_cov` = False to specify `DIP-VAE-I` model, otherwise 
        `DIP-VAE-II` model
        """

        super(DipVAE, self).__init__(
            in_channels, hidden_channels, latent_dim, input_size, **kws
        )
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.use_full_cov = use_full_cov
		
        # matrix of hyper-parameters
        param_d = torch.full((self.latent_dim, ), self.lambda_d)
        mat_d = torch.diag(param_d) # penalty on diagnal elements
        mat_od = (1 - torch.diag(torch.full((self.latent_dim, ), 1))) * self.lambda_od # penalty on off-diagnal elements
        mat_params = mat_d + mat_od

        self.mat_params = nn.Parameter(mat_params, requires_grad = False)


    def get_loss(self, original_x, reconst, mu, logvar, latent,
            *args, **kws):

        device = original_x.device
        # DIP-VAE-I use mean covariance loss only
        mat_identiy = torch.diag(torch.full((self.latent_dim, ), 1)).to(device)
        mu_0 = mu.T 
        normed_mu = mu_0 - torch.mean(mu_0, dim = 0)
        cov = torch.matmul(normed_mu, normed_mu.T)

        # DIP-VAE-II use additional variance matrix
        if self.use_full_cov:
            cov = cov + torch.diag(torch.mean(logvar.exp(), dim = 0)).to(device)
        # DIP-VAE aims to push the covariance matrix of mean vector to an identity matrix.
        cov_loss = torch.sum((cov - mat_identiy) ** 2 * self.mat_params)
        # total loss
        losses, recorder = super().get_loss(original_x, reconst, mu, logvar, latent, *args, **kws)

        total_loss = losses["reconst_loss"] + losses["kl_loss"] + cov_loss
        losses["total_loss"] = total_loss
        losses["cov_loss"] = cov_loss

        return losses, recorder
