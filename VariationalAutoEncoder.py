import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ModelUtils import *



# Loss functions
def basic_reconstruction_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum') / inputs.size(0) # normalise


cached_weights = {}

# Weight the samples over time: it's critical to get start of the sound correct.
def weighted_time_reconstruction_loss(inputs, outputs, weight, time_steps=None, verbose=True):
    assert inputs.dim() >= 2, f"Expected inputs to be greater than 2, not {inputs.dim()}"
    assert weight >= 1, f"Expected weight to be greater than 1, not {weight}"

    batch_size = inputs.size(0)
    sequence_length = inputs.size(-1)

    if inputs.dim() == 3:
        features = inputs.size(1)
    else:
        features = 1

    assert features <= sequence_length, f"Incorrect order? sequence_length={sequence_length}, features={features}"

    # Linear interpolation of weights from 'weight' to 1 over N steps
    global cached_weights
    if time_steps is None:
        time_steps = sequence_length // 10

    cached_key = (sequence_length, time_steps, weight)
    weights = cached_weights.get(cached_key)
    if weights is None:
        if time_steps > 0:
            weights = torch.linspace(weight, 1, time_steps, device=inputs.device)
            if sequence_length > time_steps:
                weights = torch.cat((weights, torch.ones(sequence_length - time_steps, device=inputs.device)))
        else:
            weights = torch.ones(sequence_length, device=inputs.device)
        cached_weights[cached_key] = weights
        if verbose:
            print(f"weight={weight}, length={sequence_length}, time_steps={time_steps}, weights={weights}")

    loss = F.mse_loss(inputs, outputs, reduction='none')

    if inputs.dim() == 3:
        loss = torch.sum(loss, dim=1)

    # Apply weights to the loss
    weighted_loss = loss * weights

    # Scale by the sum of weights
    total_weight = weights.sum()
    loss = torch.sum(weighted_loss) / (total_weight * batch_size)

    return torch.clamp(loss, 0)



def reconstruction_loss(inputs, outputs):
    assert inputs.shape == outputs.shape, f"reconstruction_loss: shapes don't match, inputs={inputs.shape}, outputs={outputs.shape}"
    return basic_reconstruction_loss(inputs, outputs)

    # I'm not too sure about this...
    return weighted_time_reconstruction_loss(inputs, outputs, weight=10)

# Test the basic loss & weighted loss:
if __name__ == '__main__':
    inputs = torch.randn(7, 20, 10)
    outputs = inputs + torch.randn(inputs.shape)*0.1
    loss1 = basic_reconstruction_loss(inputs, outputs).item()
    loss2 = weighted_time_reconstruction_loss(inputs, outputs, 1).item()
    loss3 = weighted_time_reconstruction_loss(inputs, outputs, 10, verbose=True).item()
    print(f"base: {loss1:.2f}, 1-weight: {loss2:.2f}, 10-weight: {loss3:.2f}")
    assert(abs(loss1 - loss2) < 1e-5)
    assert(abs(loss1 - loss3) < 0.7) # we expect these two to be commensurate


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
        sizes = copy.deepcopy(sizes) # we modify the first element
        super(VariationalAutoEncoder, self).__init__()
        print(f"VAE.init: sizes={sizes}")
        self.input_shape = make_torch_size(sizes[0]) # can be a single digit or a list
        sizes[0] = total_elements(sizes[0])
        print(f"VAE: input shape={self.input_shape}, size={sizes[0]} values")

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
        assert x[0].shape == self.input_shape, f"VAE.encode expected shape={self.input_shape} but got {x[0].shape}"
        x = x.view(x.size(0), -1) # flatten

        if len(self.encoder_layers):
            x = self.encoder_layers(x)
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


    def decode(self, z):
        x = self.decoder_layers(z)
        assert x.size(1) == total_elements(self.input_shape), f"VAE.decode expected {total_elements(self.input_shape)} outputs, but got {x.size(1)}"

        x = x.view((x.size(0), *self.input_shape)) # re-inflate
        #debug("decode.x", x)

        return x


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
        #debug("hiddens", hiddens)

        mu, logvar = self.vae.encode(hiddens)
        return mu, logvar


    def decode(self, z):
        hiddens = self.vae.decode(z)
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



if __name__ == "__main__":
    for sizes in ([10, 5, 2], [[4, 6], 4, 3]):
        print(f"\n\nVAE: testing sizes={sizes}")
        vae = VariationalAutoEncoder(sizes)
        batch = 7
        input_shape = make_shape_list(sizes[0])
        print(f"input_shape={input_shape}")
        inputs = torch.randn([batch] + input_shape)
        debug("inputs", inputs)
        loss, outputs = vae.forward_loss(inputs)
        assert inputs.shape == outputs.shape, f"Inputs.size={inputs.shape} vs outputs.size={outputs.shape}"
