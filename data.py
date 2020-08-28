import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import MNIST

import skimage.transform as st

import h5py

test_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"
test_npz_root = "/home/victorchen/workspace/Venus/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
test_mnist_root = "/home/victorchen/workspace/Venus/torch_download"

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

    def __len__(self):
        if not hasattr(self, "size"):
            with h5py.File(self.root, "r") as f:
                self.size = f["imgs"].shape[0]
        return self.size

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.size, "idx {} out of range".format(idx)
        with h5py.File(self.root, "r") as f:
            images = f["imgs"][idx, :]
            # images = f["imgs"][idx, :] * 255 # 因为读入的其实是二值图像，这里乘以 255 变成 uint8 范围的图像

        if self.transform:
            images = self.transform(images)
        return images

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
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.files = np.load(root)
        self.images = self.files["imgs"]
        self.size = self.images.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        assert idx >= 0 and idx < self.__len__(), "idx {} out of range".format(idx)
        image = self.images[idx, :]
        if self.transform:
            image = self.transform(image)
        return image


class ResizeArrImage(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else: self.size = (size, size)

    def __call__(self, images):
        return st.resize(images, self.size)

def get_dspritesh5py_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = dSprites_h5py(root, transform=T.Compose([
        ResizeArrImage(resize), 
        T.ToTensor()
    ]))

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_dspritesnpz_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = DSprites_npz(root, transform=T.Compose([
        ResizeArrImage(resize), 
        T.ToTensor()
    ]))

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_mnist_dataloader(root, batch_size, num_workers, resize = 32):
    trainset = MNIST(root, train = True, transform=T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,)),
    ]))
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def test_and_show_images():

    import matplotlib.pyplot as plt
    from utils import grid_plot, show_images, TensorImageUtils


    def show_preprocess(images_tensor):
        # 因为是二值图像，所以在显示和保存的时候，要还原到255.0
        return images_tensor * 255.0

    # loader = get_dsprites_dataloader(test_npz_root, 8, 4, 64)
    # print(type(loader))

    # utiler = TensorImageUtils(normalize=False, preprocess_func = show_preprocess)

    loader = get_mnist_dataloader(test_mnist_root, 8, 4, 64)
    utiler = TensorImageUtils(normalize=True)

    for i, batch in enumerate(loader):
        batch = batch[0]
        print(batch.size())
        print(batch.sum())
        utiler.plot_show(batch)
        plt.show()
        break

if __name__ == "__main__":
    test_and_show_images()
