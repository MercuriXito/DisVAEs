import os, sys, time, json
from torchvision.utils import make_grid, save_image
from torch import save, load
from math import ceil

import matplotlib.pyplot as plt

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch

def test_and_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def currentTime():
    return time.strftime("%H_%M_%S", time.localtime())

def test_and_add_postfix_dir(root):
    seplen = len(os.sep)
    if root[-seplen:] != os.sep:
        return root + os.sep
    return root

def json_dump(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)

def json_load(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config

def save_model(model, root, filename):
    save(model.state_dict(), root + filename)

def read_config_from_yaml(yml_path, encoding="utf-8"):
    with open(yml_path, encoding=encoding) as f:
        config = load(f, Loader=Loader)
    return config

class TrainLogger:
    def __init__(self, root="."):
        pass

    def _create_folder(self):
        pass

    def save_models(self, net, name, epoch):
        pass 

    def save_config(self, opt):
        pass


class TensorImageUtils:
    """Base Class of Tensor-Image utils functions including showing and saving the result images,
    `prepreocess_tensor` function is used to preprocess images before showing and saving"""
    def __init__(self, root = ".", img_range=(-1, 1), normalize=True, preprocess_func = None):
        self.root = test_and_add_postfix_dir(root)
        self.img_range = img_range
        self.normalize = normalize
        if preprocess_func is None:
            self.preprocessor = self.preprocess_tensor
        else:
            self.preprocessor = preprocess_func

    def preprocess_tensor(self, images_tensor, *args, **kws):
        """ Default preprocessor, return tensor directly
        """
        return images_tensor

    def tensor2arr(self, images, nrow = 8):
        timages = self.preprocessor(images.detach())
        grid = make_grid(
                timages, nrow=nrow, normalize=self.normalize, range=self.img_range).cpu().detach().numpy().transpose((1,2,0))
        return grid

    def plot_show(self, images, nrow = 8, figsize=(15, 15), is_rgb=False):
        fig = plt.figure(figsize=figsize)
        target_cmap = plt.cm.rainbow if is_rgb else plt.cm.binary_r
        arr = self.tensor2arr(images, nrow)
        plt.imshow(arr, cmap=target_cmap)

    def save_images(self, images, filename, nrow=8):
        images = self.preprocessor(images)
        save_image(images, self.root + filename,
                   nrow=nrow, normalize=self.normalize, range=self.img_range)

def partial_forward(func, z, batch_size, **kws):
    """ parially foward `func`, and use `batch_size` number of input in forward
    for avoiding excceeding the gpu memory.
    """
    total_num = z.size(0)
    num = 0
    items = []
    while(num < total_num):
        part_z = z[num * batch_size: min(total_num, (num + 1) * batch_size), :]
        if part_z.size(0) == 0: break
        with torch.no_grad():
            part_item = func(z, **kws)
        items.append(part_item)
        num += 1
    if len(items) == 1:
        return items[0]
    else:
        return torch.cat(items, dim=0)

if __name__ == "__main__":
    
    import numpy as np
    arr = np.random.randn(8, 32, 32, 3)
    fig = grid_plot(arr, 4)
    plt.show()
