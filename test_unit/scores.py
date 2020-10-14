import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import numpy as np

from opt import _MetaOptions
from utils import partial_forward

from opt import get_model
from data import get_loader
import os, argparse

from utils import test_and_add_postfix_dir, json_load, test_and_make_dir
from metrics.betavae_scores import BetaVAEScore

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str, help="training output folder")
    parser.add_argument("-L", type=int, default=100, help="number of pairs")
    parser.add_argument("-N", type=int, default=1000, help="number of samples for training low-capacity classifier")
    parser.add_argument("-n", "--run-times", type=int, default=30, help="total run times of evaluating scores")
    return parser.parse_args()

def test_scores():

    opt = get_options()
    L = opt.L
    N = opt.N
    epochs = opt.run_times
    path = test_and_add_postfix_dir(opt.folder)
    config = json_load(path + "config.json")
    opt = _MetaOptions.dict2opts(config)

    model = get_model(opt)
    data = get_loader(opt)

    model_path = path + "checkpoints" + os.sep + "vae.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print(model)

    scores = []
    for i in range(epochs):
        scorer = BetaVAEScore(model, data.dataset, L, N)
        score = scorer.report()
        score = score.item()
        scores.append(score)
        print("Validate Epoch {} - [score: {}]".format(i+1, score))

    output_dir = test_and_make_dir(path + "report" + os.sep)
    scores = np.array(scores)
    np.savetxt(output_dir + "scores.txt", scores)
    print("Average score: %2.6f" %(scores.mean()))
