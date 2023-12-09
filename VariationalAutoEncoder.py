import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *


# Loss functions
def reconstruction_loss(inputs, outputs):
    assert(inputs.shape == outputs.shape)
    return F.mse_loss(inputs, outputs, reduction='sum')


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Generic VAE composed of a number of fully connected linear layers (re-usable)
class VariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(sizes):
        encode = fully_connected_size(sizes) + fully_connected_size([sizes[-2], sizes[-1]])
        decode = fully_connected_size(sizes.reversed)
        return encode + decode
        
    def __init__(self, sizes, activation_fn=F.relu):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-2)])

        # Latent space layers (for mean and log variance)
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1])
        self.fc_logvar = nn.Linear(sizes[-2], sizes[-1])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i-1]) for i in reversed(range(1, len(sizes)))])

        # Activation function
        self.activation_fn = activation_fn

        print(f"VariationalAutoEncoder compression: {sizes[0]/sizes[-1]:.1f} x smaller")
        display(self)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation_fn(layer(x))
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        for idx, layer in enumerate(self.decoder_layers):
            z = layer(z)
            if idx < len(self.decoder_layers) - 1:
                z = self.activation_fn(z)
        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        error  = reconstruction_loss(inputs, outputs)
        kl_div = kl_divergence(mu, logvar)

        #print("error={:.1f}, kl_divergence={:.1f}, ratio={:.1f}".format(error, kl_div, kl_div/error))
        
        # The optimiser appears to be able to efficiently minimise the KL loss, so it's unneccesary to weight it vs the reconstruction loss.
        kl_weight = 1.0
        
        return (error + kl_weight * kl_div)
