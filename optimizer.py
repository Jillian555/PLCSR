import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD



def dis_loss(real,d_fake):
    dc_loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(real), real))
    dc_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(d_fake), d_fake))
    return dc_loss_real + dc_loss_fake


