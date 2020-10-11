import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseVAE

class VanillaVAE(BaseVAE):
    def __init__(self, 
            in_channels: int, 
            hidden_channels: list, 
            latent_dim: int, 
            input_size,
            **kws):

        super(VanillaVAE, self).__init__()

        if isinstance(input_size, list):
            input_size = tuple(input_size)
        elif isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        # Encoder Net
        modules = []
        for i in range(len(hidden_channels)):
            if i == 0:
                last_channel = in_channels
            else:
                last_channel = hidden_channels[i-1]
            modules.append(
                nn.Sequential(
                    nn.Conv2d(last_channel, hidden_channels[i], 3, 2, 1), 
                    nn.BatchNorm2d(hidden_channels[i]),
                    nn.LeakyReLU(0.2,True)
                )
            )
        self.eNet = nn.Sequential(*modules)

        dec_in_dim = 1
        enc_out_dims = []
        for size in input_size:
            single_size = int( size / int(pow(2, len(hidden_channels))))
            dec_in_dim = dec_in_dim * single_size
            enc_out_dims.append(single_size)
        self.enc_out_dims = tuple(enc_out_dims)

        if dec_in_dim == 0:
            raise ValueError("""Dim of Encoder ouput become 0, due to inappropriate 
        settings of input_size: {} and hidden_dim:{}""".format(input_size, hidden_channels))

        self.logvar_net = nn.Linear(dec_in_dim * hidden_channels[-1], latent_dim)
        self.mu_net = nn.Linear(dec_in_dim * hidden_channels[-1], latent_dim)

        self.latent_transform = nn.Linear(latent_dim, dec_in_dim * hidden_channels[-1])

        # Decoder Net
        hidden_channels = hidden_channels[::-1]
        modules = []
        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1], 4, 2, 1), 
                    nn.BatchNorm2d(hidden_channels[i+1]),
                    nn.LeakyReLU(0.2, True)
                )
            )

        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[-1], hidden_channels[-1], 4, 2, 1),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels[-1], in_channels, 3, 1, 1),
            nn.Tanh(),
        ))
        self.dNet = nn.Sequential(*modules)

    def encode(self, original_x):
        features = self.eNet(original_x).view(original_x.size(0), -1)
        logvar = self.logvar_net(features)
        mu = self.mu_net(features)
        return mu, logvar

    def decode(self, latent_z):
        z = self.latent_transform(latent_z).view(
            latent_z.size(0), self.hidden_channels[-1], self.enc_out_dims[0], self.enc_out_dims[1])
        reconst = self.dNet(z)
        return reconst

    def _sample_latent(self, mu, logvar):
        """ reparameterization tricks"""
        b = mu.size(0)
        return torch.randn((b, self.latent_dim), device=mu.device) *torch.exp(0.5 * logvar) + mu

    def forward(self, original_x):
        mu, logvar = self.encode(original_x)
        latent_samples = self._sample_latent(mu, logvar)
        reconst = self.decode(latent_samples)
        return reconst, mu, logvar, latent_samples

    def get_loss(self, original_x, reconst, mu, logvar, *args, **kws):
        """ Basic ELBO of VAE """

        reconst_weight = 1.0
        kl_weight = kws["M_N"]
        if "reconst_weight" in kws.keys(): # reconst_weight only used in test.
            reconst_weight = kws["reconst_weight"]

        kl_loss = torch.mean(-0.5* torch.sum(1 + logvar - mu **2 - logvar.exp(), dim = 1), dim=0) 

        # TODO: considering to add Bernoulli distribution for [0,1] valued images.
        reconst_loss = F.mse_loss(original_x, reconst, reduction="none").sum(dim=1).mean()
        loss = reconst_loss * reconst_weight +  kl_loss * kl_weight
        return {"total_loss": loss, "kl_loss": kl_loss, "reconst_loss": reconst_loss}, {"kl_weight": kl_weight}

    def infer(self, batch_size):
        """ random sample and infer """

        device = next(self.eNet.parameters()).device
        z = torch.randn((batch_size, self.latent_dim)).to(device)
        return self.decode(z)

if __name__ == "__main__":

    input_size = (64, 64)
    vae = VanillaVAE(1, [64, 128, 256], 20, input_size)

    x = torch.randn((32, 1, input_size[0], input_size[1])).to("cuda")
    vae.cuda()

    reconst, mu, logvar, z = vae(x)
    print(reconst.size())
    print(vae.get_loss(x, reconst, mu, logvar).items())
    print(vae)
