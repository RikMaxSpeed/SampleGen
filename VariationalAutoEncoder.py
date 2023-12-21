import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *



# Loss functions
def base_reconstruction_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum') / inputs.size(0) # normalise

# Weight the first few windows of the STFT
def weighted_stft_reconstruction_loss(inputs, outputs, weight, firstN):
    mse_loss = F.mse_loss(inputs, outputs, reduction='sum')
    firstN_loss = F.mse_loss(inputs[:, 0:firstN, :], outputs[:, 0:firstN, :], reduction='sum')
    total_loss = mse_loss + (weight-1) * firstN_loss
    total_loss *= inputs.size(1) / (inputs.size(1) + (weight - 1) * firstN) # adjust commensurately
    return total_loss / inputs.size(0)


def reconstruction_loss(inputs, outputs):
    return weighted_stft_reconstruction_loss(inputs, outputs, weight=10, firstN=5)

# Test the basic loss & weighted loss:
if __name__ == '__main__':
    inputs = torch.randn(7, 30, 50)
    outputs = inputs + torch.randn(inputs.shape)*0.1
    loss1 = base_reconstruction_loss(inputs, outputs).item()
    loss2 = weighted_stft_reconstruction_loss(inputs, outputs, 1, 5).item()
    loss3 = weighted_stft_reconstruction_loss(inputs, outputs, 10, 5).item()
    loss4 = reconstruction_loss(inputs, outputs)
    print(f"base: {loss1:.2f}, 1-weigth: {loss2:.2f}, 10-weight: {loss3:.2f}, check: {loss4:.2f}")
    assert(abs(loss1 - loss2) < 1e-4)
    assert(abs(loss1 - loss3) < 0.7) # we expect these two to be commensurate
    assert(loss3 == loss4)


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss_function(inputs, outputs, mu, logvar):

    error  = reconstruction_loss(inputs, outputs)
    kl_div = kl_divergence(mu, logvar) / inputs.size(0)
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
        
        
    @staticmethod
    def approx_trainable_parameters(sizes):
        encode = fully_connected_size(sizes) + fully_connected_size([sizes[-2], sizes[-1]])
        decode = fully_connected_size(VariationalAutoEncoder.decode_sizes(sizes))
        return encode + decode
        
    def __init__(self, sizes):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = sequential_fully_connected(sizes[:-1], default_activation_function)

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

    # For compatibility with the combined VAE
    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs


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
