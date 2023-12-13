import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *
from VariationalAutoEncoder import *


# Simply create a massive VAE with the entire spectogram as input
# This doesn't really work, evern with >100M parameters and has a lot of noise in the results.

class STFTVariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(freq_buckets, sequence_length, latent_size, depth, ratio):
        sizes = interpolate_layer_sizes(sequence_length * freq_buckets, latent_size, depth, ratio)
        return VariationalAutoEncoder.approx_trainable_parameters(sizes)

    def __init__(self, freq_buckets, sequence_length, latent_size, depth, ratio):
        super(STFTVariationalAutoEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.freq_buckets = freq_buckets
        sizes = interpolate_layer_sizes(sequence_length * freq_buckets, latent_size, depth, ratio)
        print(f"sizes={sizes}")
        self.vae = VariationalAutoEncoder(sizes)
        
        
    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.vae.encode(x)
        return vae_reparameterize(mu, logvar)
        

    def decode(self, x):
        x = self.vae.decode(x)
        x = x.reshape(x.size(0), self.freq_buckets, self.sequence_length)
        return x
        
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x, mu, logvar = self.vae.forward(x)
        x = x.reshape(x.size(0), self.freq_buckets, self.sequence_length)
        return x, mu, logvar
    

    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs
