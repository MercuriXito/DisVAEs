import os, sys, time, json
from torchvision.utils import make_grid, save_image
from torch import save, load
from math import ceil

import matplotlib.pyplot as plt

def test_and_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

def save_opt(root, opt):
    json_dump(opt._get_kwargs(), root + "config.json")

class Logger:
    """ Logger 一大意义在于不用写那么多 root 了。
    """
    def __init__(self, root):
        self.root = root

    def log_config(self, opt):
        pass

def grid_plot(arr, ncols=8):
    """ show batch of numpy-array, arr should be size of (NxHxW) or (NxHxWxC) for color image
    """
    cm = None
    if len(arr.shape) == 3:
        cm = plt.cm.binary_r
    elif len(arr.shape) == 4:
        cm = plt.cm.rainbow
    else:
        raise Exception("Size of arr {} should be 3 or 4".format(len(arr.shape)))

    bs = arr.shape[0]
    fig = plt.figure()
    nrows = ceil(bs / ncols)
    for i in range(bs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(arr[i,:], cmap = cm)
    return fig

def save_images(images, root, filename, nrow = 8):
    save_image(images, root + filename, nrow=nrow, normalize=True, range=(-1,1))

def save_model(model, root, filename):
    save(model.state_dict(), root + filename)

def show_images(images, nrow = 8, img_range = (-1, 1)):
    grid = make_grid(
                    images, nrow=nrow, range=img_range
                ).cpu().detach().numpy().transpose((1,2,0))
    return grid

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


if __name__ == "__main__":
    
    import numpy as np
    arr = np.random.randn(8, 32, 32, 3)
    fig = grid_plot(arr, 4)
    plt.show()