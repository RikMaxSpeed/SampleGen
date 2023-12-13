import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *



# Loss functions


def reconstruction_loss(inputs, outputs):
    # shape = [batch, stft, time-step]
    scale = inputs.size(1) * inputs.size(2) # data-count irrespective of batch-size
    assert(inputs.shape == outputs.shape)
    return scale * F.mse_loss(inputs, outputs)


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss_function(inputs, outputs, mu, logvar):

    error  = reconstruction_loss(inputs, outputs)
    kl_div = kl_divergence(mu, logvar)
            
    loss = error + kl_div
    #print(f"loss={loss:.2f} <-- reconstruction={error:.2f} + kl_divergence={kl_div:.2f}")

    return loss
    

def vae_reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# Generic VAE composed of a number of fully connected linear layers (re-usable)
class VariationalAutoEncoder(nn.Module):
    @staticmethod
    def decode_sizes(encoder_sizes):
        d_sizes = list(reversed(encoder_sizes))
        return d_sizes
        
        # Experiment: add an extra layer, it didn't improve the accuracy
        d_sizes = d_sizes + [d_sizes[-1]] # add an extra layer
        print(f"Encoder={encoder_sizes}, decoder sizes={d_sizes}")
        return d_sizes
        
        
    @staticmethod
    def approx_trainable_parameters(sizes):
        encode = fully_connected_size(sizes) + fully_connected_size([sizes[-2], sizes[-1]])
        decode = fully_connected_size(VariationalAutoEncoder.decode_sizes(sizes))
        return encode + decode
        
    def __init__(self, sizes):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = sequential_fully_connected(sizes[:-1], nn.ReLU())

        # Latent space layers (for mean and log variance)
        self.fc_mu     = nn.Linear(sizes[-2], sizes[-1])
        self.fc_logvar = nn.Linear(sizes[-2], sizes[-1])

        # Decoder layers
        d_sizes = VariationalAutoEncoder.decode_sizes(sizes)
        self.decoder_layers = sequential_fully_connected(d_sizes, None)

        print(f"VariationalAutoEncoder: layers={sizes}, parameters={count_trainable_parameters(self):,}, compression={sizes[0]/sizes[-1]:.1f}")

    def encode(self, x):
        if len(self.encoder_layers):
            x = self.encoder_layers(x)
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


    def decode(self, z):
        return self.decoder_layers(z)
        

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = vae_reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
        

#########################################################################################################################
# Combined_VAE: take a standard auto-encoder and insert a VAE in the middle.
# We can then train the outer auto-encoder independently, and then train the internal VAE whilst freezing the outer layers.

class CombinedVAE(nn.Module):
        
    def __init__(self, auto_encoder, sizes):
        super(CombinedVAE, self).__init__()
        
        self.auto_encoder = auto_encoder
        self.hidden_size = sizes[0]
        self.latent_size = sizes[-1]
        
        self.vae = VariationalAutoEncoder(sizes)

        print(f"CombinedVAE {count_trainable_parameters(self):,} parameters, compression={sizes[0]/sizes[-1]:.1f}")


    def encode(self, x):
        hiddens = self.auto_encoder.encode(x)
        mu, logvar = self.vae.encode(hiddens)
        return mu, logvar


    def decode(self, z):
        hiddens = self.vae.decode(z)
        stft = self.auto_encoder.decode(hiddens)
        return stft


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = vae_reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs
