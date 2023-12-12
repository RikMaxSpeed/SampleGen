import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *

zero_weight = None


# Loss functions
def reconstruction_loss(inputs, outputs):
    assert(inputs.shape == outputs.shape)
    scale = inputs.size(1) * inputs.size(2) # data-count irrespective of batch-size
    
    # Option to (heavily) weight time-step 0
    if zero_weight is None:
        return scale * F.mse_loss(inputs, outputs)
        
    # shape = [batch, stft, time-step]
    loss = (outputs - inputs) ** 2
    loss[:, :, 0] *= zero_weight
    adjust = inputs.size(2) / (inputs.size(2) + zero_weight -1)
    return scale * adjust * loss.mean()


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Generic VAE composed of a number of fully connected linear layers (re-usable)
class VariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(sizes):
        encode = fully_connected_size(sizes) + fully_connected_size([sizes[-2], sizes[-1]])
        decode = fully_connected_size(list(reversed(sizes)))
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

        print(f"VariationalAutoEncoder {count_trainable_parameters(self):,} parameters, compression={sizes[0]/sizes[-1]:.1f}")


    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation_fn(layer(x))
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu, logvar):
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
        z = reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        error  = reconstruction_loss(inputs, outputs)
        kl_div = kl_divergence(mu, logvar)
                
        #print(f"error={error:.1f}, kl_divergence={kl_div:.1f}, ratio={error/kl_div:.1f}")
        
        # The optimiser appears to always efficiently minimise the KL loss, no need to weight this.
        return error + kl_div


#########################################################################################################################
# Combined_VAE: take a standard auto-encoder and insert a VAE in the middle.
# We can then train the auto-encoder independently, and then train the VAE around that.


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
        z = VariationalAutoEncoder.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)
        

    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = self.loss_function(inputs, outputs, mus, logvars)
        return loss, outputs


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)

