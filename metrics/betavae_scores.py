"""
Disentanglement Score in BetaVAE.

+ [x] Construct Dataset with ground-truth factors,
        sample images with one specified ground-truth factors, the other remaining different.
+ [x] FC Classifier
+ [ ] Controller:
    1. N (N=10) models with same hyperparameters settings.
    2. T (T=3) times of evaluation on each model
    3. rate (rate=0.5) of dropped bottom scores amid all models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision.transforms as T

import numpy as np
import time

from opt import _MetaOptions
from utils import partial_forward

from tqdm import tqdm

class FCClassifier(nn.Module):
    """ one-layer FC Classifier for predicting true factor
    """
    def __init__(self, in_feature, num_class):
        super(FCClassifier, self).__init__()
        self.net = [nn.Linear(in_feature, num_class)]
        self.net = nn.Sequential(
            nn.Linear(in_feature, num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class ArrayDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.y[idx, :]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y


def train_val_test_split(train, val_size=0.2, test_size=0.1, shuffle=True, shuffle_seed=1):
    """ return subsetsampler of trainset, valset, testset
    """

    # assert isinstance(train, Dataset), \
    #     "Split only support on class or subclass of torch.utils.data.Dataset"

    datasize = len(train)
    indices = list(range(datasize))

    other_size = int(np.floor((val_size + test_size) * datasize))
    test_size = int(np.floor(test_size * datasize))

    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)

    train_indices, other_indices = indices[other_size:], indices[:other_size]
    test_indices, val_indices = other_indices[:test_size], other_indices[test_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, val_sampler, test_sampler


class BetaVAEScore:
    def __init__(self, 
            model, 
            dataset: torch.utils.data.Dataset, 
            L: int, # L samples
            N: int, # the size of whole constructed dataset
            ):

        self.model = model
        self.dataset = dataset
        self.L = L
        self.N = N
        self.num_latents = 5
        self.latent_sizes = dataset.latent_sizes
        self.latent_bases = np.concatenate((self.latent_sizes[::-1].cumprod()[::-1][1:],
            np.array([1,])))
        self.partial_forward_batch_size = 256
        self.latent_dim = model.latent_dim

    def create_dataset(self):

        """
        betavae score:
        1. sample y (y = 1, .., 5) ( excluding color in original dsprites)
        for b in range(B):
            2. conditionally sample L pairs of images with the same `y`
            3. encode the images with model
            4. caculate z_{diff} for each pair
            5. average diff: z_{diff}^{b}
        6. get dataset { z_{diff}^{b}, y } 
        8. divide train, validate, test part of dataset
        9. use the dataset to train classifier on train set.
        10. report accuracy on test set as the scores.
        """

        net_forward = lambda z: self.model.encode(z)[0]

        # this process is still not very efficient
        # we could sample all the images with the same y
        # and then permute the sample

        M = self.N * self.L * 2
        ys = torch.randint(0, self.num_latents, (self.N, 1)) # color is not used
        Xs = []
        for y in tqdm(ys):
            # sample 2L images for each y in ys
            latents = self.sample_latent(self.L * 2)
            latents[:, y+1] = 0 # TODO: always assign 0 is not good?
            images = self.dataset.images[self.latent_to_index(latents)]
            images = torch.tensor(images, dtype=torch.float32).unsqueeze(dim=1)

            # encode
            with torch.no_grad():
                latents = partial_forward(net_forward, images, self.partial_forward_batch_size)
            latent1, latent2 = latents.view(2, self.L, -1)
            diff = torch.abs(latent1 - latent2).mean(dim=0)
            diff = diff.detach().cpu().numpy().reshape(1, -1)
            Xs.append(diff)
        Xs = np.concatenate(Xs, axis=0)
        ys = ys.detach().cpu().numpy().reshape(-1, 1)
        Xs = torch.tensor(Xs, dtype=torch.float32)
        ys = torch.tensor(ys, dtype=torch.long)
        self.created_dataset = [Xs, ys]

    def latent_to_index(self, latents):
        return np.dot(latents, self.latent_bases).astype(int)

    def sample_latent(self, size):
        samples = np.zeros((size, self.latent_sizes.size))
        for lat_i, lat_size in enumerate(self.latent_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def report(self):

        print("creating dataset")
        self.create_dataset()
        Xs, ys = self.created_dataset
        dataset = ArrayDataset(Xs, ys)

        print("Training low capacity classifier")
        classifier = FCClassifier(self.latent_dim, self.num_latents)
        trainer = ClassifierTrainer(dataset, classifier)
        trainer.train()
        score = trainer.test_validate()
        return score


class ClassifierTrainer:
    """ simple classifier trainer """
    def __init__(self, 
            dataset,
            model,
            epochs=100,
            lr=0.01):

        self.epochs = epochs
        self.lr = lr

        self.model = model
        self.dataset = dataset
        self.val_size = 0.2
        self.test_size = 0.1
        self.train_sampler, self.val_sampler, self.test_sampler = train_val_test_split(
                dataset, val_size=self.val_size, test_size=self.test_size)
        self.val_size *= len(dataset)
        self.test_size *= len(dataset)

        self.batch_size = 256
        self.num_workers = 4

    def _init_train(self):
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self._init_train()
        loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")
        starttime = time.clock()
        for epoch in range(self.epochs):
            print("Epoch: {}/{}".format(epoch + 1, self.epochs))
            bar = tqdm(loader)
            for i, (X, y) in enumerate(bar):
                self.optimizer.zero_grad()
                ys = self.model(X)
                y = y.view(-1)
                loss = self.criterion(ys, y)
                loss.backward()
                self.optimizer.step()
                bar.set_description("[loss: %3.8f ]" %(loss.item()))
            print("Validating")
            self.validate(self.get_dataloader("val"), self.val_size)
        endtime = time.clock()
        consume_time = endtime - starttime
        print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))

    def validate(self, loader, loader_size):
        training = self.model.training
        self.model.eval()
        correct = 0
        total = loader_size
        for i, (X, y) in enumerate(loader):
            ys = self.model(X)
            predict = torch.argmax(ys, dim=1)
            y = y.view(-1,)
            correct += (predict == y).sum()
        accuracy = correct / float(total)
        print("Accuracy: {}".format(accuracy))
        self.model.train(training)
        return accuracy

    def test_validate(self):
        return self.validate(
            self.get_dataloader("all"), len(self.dataset)
        )

    def get_dataloader(self, mode):
        sampler = None
        if mode == "all":
            return DataLoader(self.dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True)

        if mode == "train":
            sampler = self.train_sampler
        elif mode == "val":
            sampler = self.val_sampler
        elif mode == "test":
            sampler = self.test_sampler
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=self.num_workers, sampler=sampler)
