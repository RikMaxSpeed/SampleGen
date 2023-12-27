import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils import *



# Loss functions
def basic_reconstruction_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum') / inputs.size(0) # normalise

# Weight the STFT over time: it's critical to get the first windows right.
def weighted_stft_reconstruction_loss(inputs, outputs, weight, slope=0.5, verbose=False):
    assert(weight >= 1)
    batch_size, sequence_length, _ = inputs.shape
    time_steps = torch.arange(sequence_length, dtype=torch.float32, device=inputs.device)

    weights = 1 + (weight - 1) * torch.exp(-slope * time_steps)
    if verbose:
        print(f"weights={weights}")

    scale = torch.sum(weights)
    loss = F.mse_loss(inputs, outputs, reduction='none')
    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=0)
    loss *= weights
    loss = loss.sum() * sequence_length / (scale * batch_size)

    # We have a bug :(
    assert loss >= 0, f"Negative loss={loss:.2f} in weighted_stft_reconstruction_loss, weight={weight:.2f}"

    return loss


def reconstruction_loss(inputs, outputs):
    assert inputs.shape == outputs.shape, f"reconstruction_loss: shapes don't match, inputs={inputs.shape}, outputs={outputs.shape}"
    return basic_reconstruction_loss(inputs, outputs)

    # TODO: fix the bug!
    if inputs.dim() == 3:
        return weighted_stft_reconstruction_loss(inputs, outputs, weight=10)
    else:
        return basic_reconstruction_loss(inputs, outputs)

# Test the basic loss & weighted loss:
if __name__ == '__main__':
    inputs = torch.randn(7, 10, 20)
    outputs = inputs + torch.randn(inputs.shape)*0.1
    loss1 = basic_reconstruction_loss(inputs, outputs).item()
    loss2 = weighted_stft_reconstruction_loss(inputs, outputs, 1).item()
    loss3 = weighted_stft_reconstruction_loss(inputs, outputs, 10, verbose=True).item()
    loss4 = reconstruction_loss(inputs, outputs)
    print(f"base: {loss1:.2f}, 1-weight: {loss2:.2f}, 10-weight: {loss3:.2f}, check: {loss4:.2f}")
    assert(abs(loss1 - loss2) < 1e-5)
    assert(abs(loss1 - loss3) < 0.7) # we expect these two to be commensurate
    assert(loss3 == loss4)


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss_function(inputs, outputs, mu, logvar):
    error  = reconstruction_loss(inputs, outputs)
    kl_div = kl_divergence(mu, logvar) / inputs.size(0)
    loss = error + kl_div

    if loss < 0:
        print(f"Negative loss!! loss={loss} (reconstruction={error}, kl_divergence={kl_div}) in vae_loss_function")
        assert loss > -1e-3, "doesn't appear to be a floating point precision problem :("
        loss = 0.0 # assume floating point discrepancy

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
        self.compression = sizes[0]/sizes[-1]
        print(f"VariationalAutoEncoder: layers={sizes}, parameters={count_trainable_parameters(self):,}, compression={self.compression:.1f}")

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
        self.compression = self.auto_encoder.compression * self.vae.compression
        print(f"CombinedVAE {count_trainable_parameters(self):,} parameters, compression={self.compression:.1f}")


    def encode(self, x):
        hiddens = self.auto_encoder.encode(x)
        assert(hiddens.size(1) == self.hidden_size)

        mu, logvar = self.vae.encode(hiddens)
        return mu, logvar


    def decode(self, z):
        hiddens = self.vae.decode(z)
        assert (hiddens.size(1) == self.hidden_size)
        sample = self.auto_encoder.decode(hiddens)
        return sample


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = vae_reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs
