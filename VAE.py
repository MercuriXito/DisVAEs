import os, sys, time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from data import get_mnist_dataloader, get_dspritesnpz_dataloader
from utils import save_images, save_model, test_and_add_postfix_dir, test_and_make_dir, currentTime, TensorImageUtils

class dSpritesImageUtils(TensorImageUtils):
    def preprocess_tensor(self, images_tensor):
        return images_tensor * 255.0

#----------------- Configuration --------------------
notebook = False
if notebook:
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


# hyper-parameter
# data_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"
save_root = "save_" + currentTime()
save_root = test_and_add_postfix_dir(save_root)
test_and_make_dir(save_root)

data_name = "mnist"

if data_name == "mnist":
    data_root = "/home/victorchen/workspace/Venus/torch_download"
    input_size = (32, 32)
    utiler = TensorImageUtils(save_root, normalize=True)
elif data_name == "dsprites":
    data_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    input_size = (64, 64)
    utiler = dSpritesImageUtils(save_root, normalize=False)

dim_z = 20
# input_size = (32, 32)
# input_size = (64, 64)
input_t = input_size[0] * input_size[1]

num_encoder_units = [input_t, 400, 200, 100]
num_decoder_units = [dim_z, 100, 200, 400, input_t]

# num_encoder_units = [input_t, 100]
# num_decoder_units = [dim_z, input_t]

batch_size = 256
grid_row = 16
num_workers = 4 # 使用多线程进行 preprocess 现在好像还有问题
epochs = 50
lr = 0.0001
momentum = 0.9
cuda = True

num_iter_save_model = 2
test_training = False # set False to start real training. test_training would train for one epoch

if data_name == "mnist":
    data = get_mnist_dataloader(data_root, batch_size, num_workers, resize=input_size)
elif data_name == "dsprites":
    data = get_dspritesnpz_dataloader(data_root, batch_size, num_workers, resize = input_size)

#----------------- Model Architecture -------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        num_units = num_encoder_units
        self.net = []
        for i in range(len(num_units) - 1):
            self.net += [
                nn.Linear(num_units[i], num_units[i+1]),
                nn.ReLU(True),
            ]
        self.net = nn.Sequential(*self.net)
        self.net_logsigma = nn.Linear(num_units[-1], dim_z)
        self.net_mu = nn.Linear(num_units[-1], dim_z)

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.net(x)
        return self.net_mu(x), self.net_logsigma(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        num_units = num_decoder_units
        self.net = []
        for i in range(len(num_units) - 2):
            self.net += [
                nn.Linear(num_units[i], num_units[i+1]),
                # nn.ReLU(True),
                nn.LeakyReLU(0.2)
            ]
        self.net += [nn.Linear(num_units[-2], num_units[-1])] # No relu at last layer
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class VAE(nn.Module):
    def __init__(self, encoder , decoder):
        super(VAE, self).__init__()
        self.enc = encoder
        self.dec = decoder

    def encode(self, x):
        batch_size, c, h, w = x.size()
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.enc(x)
        # reparameterization trick
        w = torch.randn((batch_size, dim_z), device=x.device)
        y = w * torch.exp(log_sigma) + mu
        return mu, log_sigma, y

    def decode(self, z):
        reconst = self.dec(z)
        return reconst

    def forward(self, x):
        batch_size, c, h, w = x.size()
        mu, log_sigma, z = self.encode(x)
        reconst = self.decode(z).view(batch_size, c, h, w)
        return mu, log_sigma, reconst, z
    
    def KLLoss(self, mu, log_sigma):
        return torch.mean(-0.5* torch.sum(1 + log_sigma - mu **2 - log_sigma.exp(), dim = 1), dim=0)

#----------------- Data, Models definition --------------------

writer = SummaryWriter(save_root)

encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder)

# data = get_dsprites_dataloader(data_root, batch_size, num_workers, resize = input_size)

if cuda:
    encoder.cuda()
    decoder.cuda()
    vae.cuda()

# recons loss
mse = nn.MSELoss(reduction="sum")

# optimizer_enc = optim.SGD(encoder.parameters(), lr = lr, momentum=momentum)
# optimizer_dec = optim.SGD(decoder.parameters(), lr = lr, momentum=momentum)

optimizer_enc = optim.Adam(encoder.parameters(), lr = lr)
optimizer_dec = optim.Adam(decoder.parameters(), lr = lr)

lr_scheduler_enc = optim.lr_scheduler.StepLR(optimizer_enc, 10, gamma=0.5)
lr_scheduler_dec = optim.lr_scheduler.StepLR(optimizer_dec, 10, gamma=0.5)

iter_step = 0

kl_weight = 10
#----------------- Training -------------------
starttime = time.clock()
for epoch in range(epochs):
    print("Epoch:{}/{}".format(epoch+1, epochs))
    for i, batches in enumerate(tqdm(data)):
        if data_name == "mnist":
            images, _ = batches
        else:
            images = batches
        images = images.type(torch.float32)
        if cuda:
            images = images.cuda()

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        # forward and loss
        mu, log_sigma, reconst, z = vae(images)
        reconst_loss = mse(images, reconst)
        kl_loss = vae.KLLoss(mu, log_sigma)
        loss = reconst_loss + kl_loss * kl_weight

        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()

        # log
        iter_step += 1
        writer.add_scalar("reconst_loss", reconst_loss.item(), iter_step)
        writer.add_scalar("kl_loss", kl_loss.item(), iter_step)
        writer.add_scalar("loss", loss.item(), iter_step)
    
    lr_scheduler_dec.step()
    lr_scheduler_enc.step()

    if epoch % num_iter_save_model == 0 or epoch == epochs - 1 or test_training:
        images = images.view(-1, 1, input_size[0], input_size[1])
        reconst = reconst.view(-1, 1, input_size[0], input_size[1])
        save_model(encoder, save_root, "encoder.pt")
        save_model(decoder, save_root, "decoder.pt")
        utiler.save_images(images,"epoch_{}_true.png".format(epoch), nrow=grid_row)
        utiler.save_images(reconst, "epoch_{}_fake.png".format(epoch), nrow=grid_row)

    if test_training:
        break

endtime = time.clock()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))