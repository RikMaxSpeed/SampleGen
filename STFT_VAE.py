import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *
from VariationalAutoEncoder import *


# Simply create a massive VAE with the entire spectogram as input
# This does work but requires >100M parameters and has a lot of noise in the results.

class STFTVariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(stft_buckets, sequence_length, latent_size, depth, ratio):
        sizes = interpolate_layer_sizes(sequence_length * stft_buckets, latent_size, depth, ratio)
        return VariationalAutoEncoder.approx_trainable_parameters(sizes)

    def __init__(self, stft_buckets, sequence_length, latent_size, depth, ratio):
        super(STFTVariationalAutoEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.stft_buckets = stft_buckets
        sizes = interpolate_layer_sizes(sequence_length * stft_buckets, latent_size, depth, ratio)
        self.vae = VariationalAutoEncoder(sizes, nn.Tanh)
        
        
    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.vae.encode(x)
        return vae_reparameterize(mu, logvar)
        

    def decode(self, x):
        x = self.vae.decode(x)
        x = x.reshape(x.size(0), self.stft_buckets, self.sequence_length)
        return x
        
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x, mu, logvar = self.vae.forward(x)
        x = x.reshape(x.size(0), self.stft_buckets, self.sequence_length)
        return x, mu, logvar
    

    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs
