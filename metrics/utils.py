import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision.transforms as T

import numpy as np
import time

from tqdm import tqdm, trange

class FCClassifier(nn.Module):
    """ low capacity classifier with only one dense layer for predicting true factor
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


class TensorDataset(Dataset):
    """ Dataset for loading X and y of type `torch.Tensor`
    """
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.y[idx, :]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y

class ArrayDataset(TensorDataset):
    """ Dataset for loading X and y of type `np.ndarray`
    """
    def __init__(self, X, y, transform=None, target_transform=None):
        assert X.shape[0] == y.shape[0]
        if transform is None:
            transform = T.Compose([T.ToTensor()])
        super().__init__(X, y, transform, target_transform)

    def __len__(self):
        return self.X.shape[0]


def train_val_test_split(train, val_size=0.2, test_size=0.1, shuffle=True, shuffle_seed=1):
    """ return subsetsampler of trainset, valset, testset
    """

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


class ClassifierTrainer:
    """ simple classifier trainer """
    def __init__(self, 
            dataset,
            model,
            epochs=200,
            lr=0.01,
            decay_step=50):

        self.epochs = epochs
        self.lr = lr
        self.decay_step = decay_step

        self.model = model
        print(model)
        self.dataset = dataset
        self.val_size = 0.2
        self.test_size = 0.1
        self.train_sampler, self.val_sampler, self.test_sampler = train_val_test_split(
                dataset, val_size=self.val_size, test_size=self.test_size)
        self.val_size *= len(dataset)
        self.test_size *= len(dataset)

        self.batch_size = 32
        self.num_workers = 4

    def _init_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.decay_step,
                gamma=0.1)

    def train(self):
        self._init_train()
        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")
        starttime = time.clock()
        for epoch in range(self.epochs):
            with tqdm(train_loader, leave=False) as t:
                for i, (X, y) in enumerate(t):
                    self.optimizer.zero_grad()
                    ys = self.model(X)
                    y = y.view(-1)
                    loss = self.criterion(ys, y)
                    loss.backward()
                    self.optimizer.step()
                    t.set_description("Epoch: %d/%d: [loss: %3.8f ]" %(epoch + 1, self.epochs, loss.item()))
            val_score = self.validate(self.get_dataloader("val"), self.val_size)
            print("Epoch: {}/{} - Validating: {}".format(epoch+1, self.epochs, val_score))
            self.lr_scheduler.step()
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
        self.model.train(training)
        return accuracy

    def test_validate(self):
        return self.validate(
            self.get_dataloader("all"), len(self.dataset) # TODO: test on test dataset?
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
