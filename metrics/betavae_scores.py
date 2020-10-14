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

from metrics.utils import FCClassifier, TensorDataset, ArrayDataset, \
        ClassifierTrainer

# TODO: currently, this score could only used on dsprites
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
        self.latent_sizes = dataset.latent_sizes
        self.num_latents = len(self.latent_sizes) - 1
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
        for i, y in enumerate(tqdm(ys)):
            # sample 2L images for each y in ys
            latents = self.sample_latent(self.L * 2)
            latent_range = self.latent_sizes[y+1] # ignore color
            latent_indices = np.random.randint(0, latent_range, (self.L, 1)).repeat(2, axis=1).reshape(-1, )
            latents[:, y+1] = latent_indices # TODO: right now I haven't promise to keep other latents different.
            images = self.dataset.images[self.latent_to_index(latents)]
            images = torch.tensor(images, dtype=torch.float32).unsqueeze(dim=1)

            # encode
            with torch.no_grad():
                latents = partial_forward(net_forward, images, self.partial_forward_batch_size)
            latent1, latent2 = latents.view(self.L, 2, -1).transpose(1,0)
            diff = torch.abs(latent1 - latent2).mean(dim=0)
            diff = diff.detach().cpu().numpy().reshape(1, -1)
            Xs.append(diff)
        Xs = np.concatenate(Xs, axis=0)
        ys = ys.detach().cpu().numpy().reshape(-1, 1)
        Xs = torch.tensor(Xs, dtype=torch.float32)
        ys = torch.tensor(ys, dtype=torch.long)
        self.created_dataset = [Xs, ys] # mind Xs and ys are not shuffled.

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
        dataset = TensorDataset(Xs, ys)

        print("Training low capacity classifier")
        classifier = FCClassifier(self.latent_dim, self.num_latents)
        # for ground-truth labels, the classifer should reach 100% accuracy
        # classifier = FCClassifier(len(self.latent_sizes)-1, self.num_latents) # for test
        trainer = ClassifierTrainer(dataset, classifier)
        trainer.train()
        score = trainer.test_validate()
        return score
