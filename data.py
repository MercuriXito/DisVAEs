import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import MNIST
import torch

import skimage.transform as st
from utils import TensorImageUtils

import h5py

test_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"
test_npz_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
test_mnist_root = "/home/victorchen/workspace/Venus/torch_download/MNIST"

# --------------------- Dataset -----------------------
class Hdf5Dataset(Dataset):
    """ Base Class for loading Hdf5Dataset, x_dataset, target_dataset are dataset object in hdf5 file.
    """
    def __init__(self, x_dataset, target_dataset, transform = None, target_transform = None):
        self.x = x_dataset
        self.y = target_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.__len__(), "idx {} out of range".format(idx)
        images, target = self.x[idx, :], self.y[idx, :]
        if self.transform:
            self.transform(images)
        if self.target_transform:
            self.target_transform(target)
        return images, target

class dSprites_h5py(Dataset):
    """ Loading dSprites dataset (h5py) from https://github.com/deepmind/dsprites-dataset, loading hdf5 cosumes much less memory than loading npzfile
    but also brings more time to accesss array randomly.
    """
    def __init__(self, root, transform = None, target_transform = None, 
        with_classes = False, with_values = False, random_seed = 1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.seed = random_seed
        self.opened = False
        self.with_classes = with_classes
        self.with_values = with_values

        # self.f = h5py.File(self.root, "r")

    def __len__(self):
        if not hasattr(self, "size"):
            with h5py.File(self.root, "r") as f:
                self.size = f["imgs"].shape[0]
        return self.size

    def __getitem__(self, idx):

        # image = self.f["imgs"][idx, :] * 255.0
        # clss = self.f["latents/classes"][idx, :] # class already one-hotted
        # value = self.f["latents/values"][idx, :]

        with h5py.File(self.root, "r") as f:
            image = f["imgs"][idx, :] * 255
            clss = f["latents/classes"][idx, :] # class already one-hotted
            value = f["latents/values"][idx, :]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            clss, value = self.target_transform(clss, value)
        return image, (clss, value)

    def close(self):
        self.f.close()

    def next_batch(self, batch_size = 64, shuffle = True):
        """ Simple Generator of Batch Loading, without transformation
        """
        np.random.seed(self.seed)
        indices = np.arange(0, self.size, 1, dtype = int)
        if shuffle:
            np.random.shuffle(indices)
        num_batches = self.size // batch_size
        for i in range(num_batches + 1):
            start = batch_size * i
            end = min(batch_size * (i + 1), self.size)
            idx = indices[start: end]
            idx.sort()
            x = self.images[idx, :] # 随机访问 hdf5 文件的性能影响比较大
            yield x

class DSprites_npz(Dataset):
    """ Loading dSprites dataset (npz) from https://github.com/deepmind/dsprites-dataset
    """
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.files = np.load(root, allow_pickle=True, encoding="latin1")
        self.images = self.files["imgs"]
        self.size = self.images.shape[0]
        self.values = self.files["latents_values"]
        self.classes = self.files["latents_classes"]

        # additional information
        self.meta = self.files["metadata"][()]
        self.latent_sizes = self.meta["latents_sizes"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        assert idx >= 0 and idx < self.__len__(), "idx {} out of range".format(idx)
        image = self.images[idx, :] * 255
        clss = self.classes[idx, :]
        value = self.values[idx, :]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            clss, value = self.target_transform(clss, value)
        return image, (clss, value) # add labels further

# --------------------- Transform -----------------------

class ResizeArrImage(object):
    """ Reize Array.
    """
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else: self.size = (size, size)

    def __call__(self, images):
        return st.resize(images, self.size)

class SingleArrToTensor:
    """ convert ndarray images with single channel to tensor
    """
    def __call__(self, image):
        return torch.from_numpy(image).unsqueeze(dim=0).type(torch.float32)

class ReNormalize:
    def __call__(self, image):
        image = image / 255.0
        return image * 2 - 1

# --------------------- Data Loader -----------------------

def get_dspritesh5py_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = dSprites_h5py(root, transform=T.Compose([
        ResizeArrImage(resize), 
        T.ToTensor()
    ]))

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_dspritesnpz_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = DSprites_npz(root, transform=T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),
    ]))

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

def get_mnist_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = MNIST(root, train = True, transform=T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),
    ]))
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def get_loader(opt):
    data_name = opt.data_name
    data_root = opt.data_path

    batch_size = opt.batch_size
    num_workers = opt.num_workers
    shuffle = opt.shuffle
    size = opt.input_size
    resize = (size, size)

    if data_name == "mnist":
        # loader, persistor
        trainset = MNIST(data_root, train = True, transform=T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize((0.5,),(0.5,)),
        ]))
    elif data_name == "dsprites":
        trainset = DSprites_npz(data_root, transform=T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize((0.5,),(0.5,)),
        ]))
    else:
        raise NotImplementedError("Not supported dataset: {}".format(data_name))

    loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_utiler(data_name, save_root="./"):

    preprocess_func = None
    normalize = True
    img_range = (-1,1)

    if data_name == "dsprites":
        pass
    elif data_name == "mnist":
        pass
    else:
        raise NotImplementedError("Not supported dataset: {}".format(data_name))

    utiler = TensorImageUtils(root=save_root, img_range=img_range, normalize=normalize, 
        preprocess_func=preprocess_func)
    return utiler

def test_loader():
    import matplotlib.pyplot as plt
    from opt import _MetaOptions

    # loader = get_mnist_dataloader(test_mnist_root, 8, 4, 64)
    loader = get_loader(_MetaOptions.kws2opts(
            data_name="dsprites",
            data_path=test_npz_root,
            batch_size=4,
            num_workers=0,
            input_size=64,
            shuffle=True,
        ))
    utiler = TensorImageUtils()

    for i, batch in enumerate(loader):
        images, (clss, values) = batch
        print(images)
        print(clss)
        print(values)
        print((images == 1).sum())
        print(images.size())
        print(images.sum())
        utiler.plot_show(images, nrow=8)
        print(loader.dataset.latent_sizes)
        plt.show()
        break
    pass

if __name__ == "__main__":
    test_loader()
