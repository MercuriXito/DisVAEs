import torch.nn as nn

class BaseVAE(nn.Module):
    def __init__(self, *args, **kws):
        super(BaseVAE, self).__init__()

    def forward(self, *args, **kws):
        raise NotImplementedError("not implemented")

    def encode(self, *args, **kws):
        raise NotImplementedError("not implemented")

    def decode(self, *args, **kws):
        raise NotImplementedError("not implemented")

    def infer(self, *args, **kws):
        raise NotImplementedError("not implemented")

    def _sample_latent(self, *args, **kws):
        raise NotImplementedError("not implemented")

    def get_loss(self, *args, **kws):
        raise NotImplementedError("not implemented")
