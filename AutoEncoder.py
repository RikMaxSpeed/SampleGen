import torch
import torch.nn as nn
import torch.nn.functional as F
from Debug import *

# Generic VAE composed of a number of fully connected linear layers (re-usable)
class VariationalAutoEncoder(nn.Module):
    def __init__(self, sizes, activation_fn=F.relu):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-2)])

        # Latent space layers (for mean and log variance)
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1])
        self.fc_logvar = nn.Linear(sizes[-2], sizes[-1])

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i-1]) for i in reversed(range(1, len(sizes)))]
        )

        # Activation function
        self.activation_fn = activation_fn


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


    def reconstruction_loss(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction='sum')
        
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    def loss_function(self, recon_x, x, mu, logvar):
        error  = self.reconstruction_loss(recon_x, x)
        kl_div = self.kl_divergence(mu, logvar)

        #print("error={:.3f}, kl_divergence={:.3f}, ratio={:.1f}".format(error, kl_div, kl_div/error))
        
        # The model is always able to efficiently minimise the KL loss, so it's unneccesary to weight it vs the reconstruction loss.
        kl_weight = 1.0
        
        return (error + kl_weight * kl_div) / x[0].numel()



class STFTVariationalAutoEncoder(nn.Module):
    def __init__(self, sequence_length, stft_buckets, sizes, activation_fn):
        super(STFTVariationalAutoEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.stft_buckets = stft_buckets
        sizes = [sequence_length * stft_buckets] + sizes
        print("STFTVariationalAutoEncoder: sequence_length={}, stft_buckets={}, sizes={}, activation_fn={}".format(sequence_length, stft_buckets, sizes, activation_fn.__class__))
        self.vae = VariationalAutoEncoder(sizes, activation_fn)
        
        
    def encode(self, x, randomize):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.vae.encode(x)
        
        if randomize:
            return self.vae.reparameterize(mu, logvar) # for generation
        else:
            return mu # for precise reproduction
        
        
    def decode(self, x):
        x = self.vae.decode(x)
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x
        
        
    def forward(self, x, randomize):
        x = x.reshape(x.size(0), -1)
        x, mu, logvar = self.vae.forward(x)
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x, mu, logvar


    def loss_function(self, recon_x, x, mu, logvar):
        return self.vae.loss_function(recon_x, x, mu, logvar)
