"""
Loss of several.

    reconst_type: { bernoulli, gaussian }
    annealing_kl: annealing of kl_loss: 
        - [ ]: anneal in the whole training ?

"""

import torch
import torch.nn.functional as F

def _reconst_loss(output, target, reconst_type):
    """ Reconstruction loss for different prior distribution. { gaussian or bernoulli }
    """
    reconst_loss = 0.0
    if reconst_type == "gaussian":
        reconst_loss = F.mse_loss(output, target, reduction="none").sum(dim=1).mean()
    elif reconst_type == "bernoulli":
        reconst_loss = F.binary_cross_entropy_with_logits(output, target, reduction="sum") / output.size(0)
    else:
        raise NotImplementedError("Reconst loss:{} not implemented".format(reconst_type))

    return reconst_loss

def _kl_divergence_loss(mu, logvar, reduce=False, kl_weight=1, sep_weight=None):
    """ KL divergenece between N(0,I) and N(mu, logvar.exp()*I)

    :params:
        reduct (bool) - sum up the kl divergence of each component or not, default False.
    """
    kl_loss = -0.5* torch.mean(1 + logvar - mu **2 - logvar.exp(), dim = 0)

    if sep_weight is not None:
        assert kl_loss.size(0) == sep_weight.size(0)
        ndim = len(kl_loss.size())
        sep_weight = sep_weight.view(-1, *[1 for i in range(ndim-1)])
        kl_loss *= sep_weight

    if reduce:
        return kl_loss.sum()
    return kl_loss * kl_weight
