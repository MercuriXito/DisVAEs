import torch 
import torch.nn as nn
import torch.nn.functional as F

class VanillaVAE(nn.Module):
    def __init__(self, 
            in_channels: int, 
            hidden_channels: list, 
            latent_dim: int, 
            input_size: tuple):

        super(VanillaVAE, self).__init__()

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
                    nn.ReLU(True)
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
                    nn.LeakyReLU(0.2, True)
                )
            )

        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[-1], hidden_channels[-1], 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels[-1], in_channels, 3, 1, 1),
            # nn.Tanh() # for other images normalized in [-1,1]
            # nn.Sigmoid(), # images in dSprites dataset is binary
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
        """ reparameterization 
        """
        b = mu.size(0)
        return torch.randn((b, self.latent_dim), device=mu.device) *torch.exp(0.5 * logvar) + mu

    def forward(self, original_x):
        mu, logvar = self.encode(original_x)
        latent_samples = self._sample_latent(mu, logvar)
        reconst = self.decode(latent_samples)
        return reconst, mu, logvar

    def get_loss(self, original_x, reconst, mu, logvar, *args, **kws):
        kl_loss = torch.mean(-0.5* torch.sum(1 + logvar - mu **2 - logvar.exp(), dim = 1), dim=0)
        reconst_loss = F.mse_loss(original_x, reconst, reduction="sum")
        reconst_loss = reconst_loss * original_x.size(0) # scale up reconst loss
        return {"total_loss": kl_loss + reconst_loss, "kl_loss": kl_loss, "reconst_loss": reconst_loss}

    def generate(self, x):
        return self.forward(x)

if __name__ == "__main__":

    input_size = (64, 64)
    vae = VanillaVAE(1, [64, 128, 256], 20, input_size)

    x = torch.randn((32, 1, input_size[0], input_size[1])).to("cuda")
    vae.cuda()

    reconst, mu, logvar = vae(x)
    print(reconst.size())
    print(vae.get_loss(x, reconst, mu, logvar).items())
    print(vae)