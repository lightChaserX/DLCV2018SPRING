import torch
import torch.nn as nn

reconstruction_function = nn.MSELoss()

def loss_function(recon_x, x, mu, logvar, lambda_kl):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = reconstruction_function(recon_x, x)
    # loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    loss = MSE + lambda_kl*KLD
    return loss, MSE, KLD