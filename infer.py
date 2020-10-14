import argparse, os

import torch
import numpy as np

from data import get_utiler
from opt import get_model, _MetaOptions
from utils import read_config_from_yaml, json_load, test_and_make_dir


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str, help="training output folder")
    parser.add_argument("--mode", default="traversal_all", type=str, help="inference mode")
    parser.add_argument("--seed", default="-1", type=int, help="random seed, seed < 0 for not using seed")
    parser.add_argument("--zrange", default="2", type=int, help="latent traversal range")
    return parser.parse_args()

def get_z(opt, mode):
    """ return zs and names """
    if mode == "interpolation":
        # random interpolation
        z1, z2 = torch.randn((2, opt.model_params.latent_dim)).type(torch.float32)
        gap = (z2 - z1).unsqueeze(dim=0)
        interpolate = torch.linspace(0, 1, opt.batch_size).unsqueeze(dim=1)
        interpolate = interpolate * gap + z1
        return [interpolate], ["sample.png"]
    elif mode == "traversal":
        # interpolate on one dimension and preserve the other dimension
        latent_dim = opt.model_params.latent_dim
        z_range = (-opt.zrange,opt.zrange)
        zs = []
        names = []
        for i in range(latent_dim):
            # use uniform samples instead of gaussian samples to avoiding large component
            # z1 = torch.randn((1, latent_dim )).repeat(opt.batch_size, 1).type(torch.float32) # gaussian
            z1 = torch.rand((1, latent_dim )).repeat(opt.batch_size, 1).type(torch.float32) * 2 - 1
            interpolate = torch.linspace(*z_range, opt.batch_size)
            z1[:, i] = interpolate
            zs.append(z1)
            names.append("traversal_dim_{}.png".format(i+1))
        return zs, names
    elif mode == "traversal_all":

        latent_dim = opt.model_params.latent_dim
        z_range = (-opt.zrange,opt.zrange)
        z_start = torch.randn((1, latent_dim)).repeat(opt.batch_size * latent_dim, 1).type(torch.float32).view(latent_dim, opt.batch_size, -1)
        # z_start = torch.full((1, latent_dim), z_range[0]).repeat(opt.batch_size * latent_dim, 1).type(torch.float32).view(latent_dim, opt.batch_size, -1)
        # traverse each seperate component
        for i in range(latent_dim):
            interpolate = torch.linspace(*z_range, opt.batch_size)
            z_start[i,:,i] = interpolate

        return [z_start.view(-1, latent_dim)], ["traversal_all.png"]
    else:
        raise NotImplementedError("Not supported inference mode: {}".format(mode))

def inference():

    # infer options
    infer_opt = get_options()
    path = infer_opt.folder
    mode = infer_opt.mode
    seed = infer_opt.seed
    zrange = infer_opt.zrange

    if seed >= 0:
        torch.manual_seed(seed)
    else:
        print(torch.initial_seed())

    # options in training
    config = json_load(path + "config.json")
    for name, key in infer_opt._get_kwargs():
        config[name] = key
    opt = _MetaOptions.dict2opts(config)

    # WARNING: delete test lines below
    opt.batch_size = 10
    opt.grid_nrow = opt.batch_size
    # opt.grid_nrow = 1
    # opt.grid_nrow = 8

    if mode == "traversal_all":
        opt.grid_nrow = opt.batch_size

    # load zs and output picture names
    zs, names = get_z(opt, mode)

    # load model
    model = path + "checkpoints" + os.sep + "vae.pt"
    vae = get_model(opt)
    vae.load_state_dict(torch.load(model, map_location="cpu"))
    vae.cuda()

    default_output_dir = test_and_make_dir(path + os.sep + "test" + os.sep) # default output directory
    utiler = get_utiler(opt.data_name, default_output_dir)
    for z, image_name in zip(zs, names):
        z = z.cuda()
        reconst = vae.decode(z)
        utiler.save_images(reconst, image_name,  nrow=opt.grid_nrow)
        print("Image saved in {}".format(image_name))
        np.savetxt(default_output_dir + "{}.txt".format(image_name), z.detach().cpu().numpy())


if __name__ == "__main__":
    inference()
