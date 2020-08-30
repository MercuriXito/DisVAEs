import os, sys, time
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_loader, get_utiler
from utils import save_model, test_and_add_postfix_dir, test_and_make_dir, currentTime
from models.vanilla_vae import VanillaVAE

#----------------- Configuration --------------------
notebook = False
if notebook:
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# hyper-parameters
save_root = "save_" + currentTime()
save_root = test_and_add_postfix_dir(save_root)
test_and_make_dir(save_root)

# read configuration from yaml
with open("./configs/vanilla_vae.yml", encoding="utf-8") as f:
    config = load(f, Loader=Loader)

print("Using Config:")
for key, val in config.items():
    print("\t{}:{}".format(key, val))
print("Saving at {}".format(save_root))

# model parameters
latent_dim = config["model_params"]["latent_dim"]
hidden_channels = config["model_params"]["hidden_channels"]
in_channels = config["model_params"]["in_channels"]

# data parameters
data_name = config["data_params"]["data_name"]
data_root = config["data_params"]["datapath"]
input_size = tuple(config["data_params"]["input_size"])
grid_nrow = config["data_params"]["grid_nrow"]

# training parameters
lr = config["training_params"]["lr"]
cuda = config["training_params"]["use_cuda"]
epochs = config["training_params"]["epochs"]
lr_decay_weight = config["training_params"]["lr_decay_weight"]
decay_step = config["training_params"]["decay_step"]
betas = config["training_params"]["betas"]

# save parameters
num_iter_save_model = config["save_params"]["num_iter_save_model"]
test_training = False # set False to start real training. test_training would train for one epoch

#----------------- Data, Models definition --------------------
data = get_loader(config)
utiler = get_utiler(data_name, save_root)

vae = VanillaVAE(in_channels, hidden_channels, latent_dim, input_size)

print(vae)

if cuda:
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr = lr, betas=tuple(betas))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, decay_step, gamma=lr_decay_weight)

writer = SummaryWriter(save_root)
iter_step = 0
#----------------- Training -------------------
print("Start Training.")
starttime = time.clock()
for epoch in range(epochs):
    print("Epoch:{}/{}".format(epoch+1, epochs))
    for i, batches in enumerate(tqdm(data)):
        images, _ = batches
        images = images.type(torch.float32)
        size = images.size(0)
        if cuda:
            images = images.cuda()

        optimizer.zero_grad()
        # forward and calculate loss
        reconst, mu, logvar = vae(images)
        losses = vae.get_loss(images, reconst, mu, logvar)
        total_loss = losses["total_loss"]

        total_loss.backward()
        optimizer.step()

        # log
        iter_step += 1
        for name, loss in losses.items():
            writer.add_scalar(name, loss.item(), iter_step)
    
    lr_scheduler.step()
    if epoch % num_iter_save_model == 0 or epoch == epochs - 1 or test_training:
        images = images.view(-1, 1, input_size[0], input_size[1])
        reconst = reconst.view(-1, 1, input_size[0], input_size[1])
        save_model(vae, save_root, "vae.pt")
        utiler.save_images(images,"epoch_{}_true.png".format(epoch), nrow=grid_nrow)
        utiler.save_images(reconst, "epoch_{}_fake.png".format(epoch), nrow=grid_nrow)

    if test_training:
        break

endtime = time.clock()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))