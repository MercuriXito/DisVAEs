import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Importance Weighted AutoEncoder 的实现：

Ways to implement IWAE:
    1. 将每个输入 x_i 复制 k 份，然后按照相同的方式通过 VAE，最后对相同的 k 份内求各自的权重，权重相加
    2. 1 其实存在比较多的重复，比如 var, mu 都只和 x_i 有关，每个 x_i 只需要 Encoder forward 一次，然后每个复制 k 份，
    采样 batchsize * k 个样本，通过 decoder.

下面的实现采用的是 way1, 可以改成 way2.
"""

# for 64x64 input
class IWAE(nn.Module):
    def __init__(self, k, dim_z, in_channels, replicate_mode=True, 
            **kws):
        super(IWAE, self).__init__()

        self.latent_dim = dim_z
        self.k = k

        # Decoder
        self.dNet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.logvarNet = nn.Sequential(
            nn.Conv2d(256, dim_z, 4, 1, 0)
        )
        self.muNet = nn.Sequential(
            nn.Conv2d(256, dim_z, 4, 1, 0)
        )

        self.latent_transform = nn.Linear(dim_z, 256)

        # Encoder
        self.eNet = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        """ Decoder: dNet + (logvarNet, muNet)
        """
        feature = self.dNet(x)
        mu = self.muNet(feature).view(feature.size(0), self.latent_dim)
        logvar = self.logvarNet(feature).view(feature.size(0), self.latent_dim)
        return mu, logvar

    def decode(self, z):
        """ Encoder: latent transform + eNet
        """
        z = self.latent_transform(z).view(z.size(0), -1, 1, 1)
        reconst = self.eNet(z)
        return reconst

    def forward(self, x):
        rex = self.replicate_x(x)
        print(rex.size())
        mu, logvar = self.encode(rex)
        z_sample = self.sample_latent(mu, logvar)
        reconst = self.decode(z_sample)
        return reconst, mu, logvar, z_sample, rex

    def sample_latent(self, mean, logvar):
        """ reparameterization trick
        """
        b = mean.size(0)
        return torch.randn((b, self.latent_dim), device=mean.device) * torch.exp(logvar * 0.5) + mean

    def prob_qhx(self, z_sample, mean, logvar):
        """ probability of z_sample in q(h|x)
        """
        y = -0.5 * (z_sample - mean) ** 2 / logvar.exp()
        return torch.exp(y.sum(dim=1))

    def replicate_x(self, x):
        """ expand any tensor to size: (x.size(0) * self.k , ... )
        """
        b = x.size(0)
        other = x.size()[1:]

        repeat_times = [1 for i in range(len(x.size()) + 1)]
        repeat_times[1] = self.k

        x.unsqueeze_(dim=1)
        x = x.repeat(*repeat_times)
        x = x.view(b * self.k, *other)
        return x

    def get_loss_k(self, original_x, reconst, mu, logvar, z_sample, *args, **kws):
        """ original_x: batch size: (original_batch_size * k)
        """

        mse = (original_x - reconst) ** 2
        reconst_loss = mse.view(mse.size(0), -1).sum(dim=1)
        kl_loss = -0.5* torch.sum(1 + logvar - mu **2 - logvar.exp(), dim=1)
        loss = reconst_loss + kl_loss
        loss = loss.view(-1, self.k)

        weight = 1 / self.prob_qhx(z_sample, mu, logvar)
        weight = weight.view(-1, self.k)
        weight_sum = weight.sum(dim=1).unsqueeze(dim=1)
        weight = weight / weight_sum

        loss = loss * weight
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return {"total_loss":loss, "kl_loss": kl_loss.mean(), "reconst_loss": reconst_loss.mean()}
