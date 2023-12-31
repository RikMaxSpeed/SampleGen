import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ModelUtils import *

percentage = 100

# Loss functions
def basic_reconstruction_loss(inputs, outputs):
    return percentage * F.mse_loss(inputs, outputs)


cached_weights = {}

# Weight the samples over time: it's critical to get start of the sound correct.
def weighted_time_reconstruction_loss(inputs, outputs, weight, time_ratio=0.15, verbose=False):
    assert inputs.dim() >= 2, f"Expected inputs to be greater than 2, not {inputs.dim()}"
    assert weight >= 1, f"Expected weight to be greater than 1, not {weight}"
    assert time_ratio >= 0 and time_ratio <= 1, f"Expected time_ratio to be between 0 and 1, not {time_ratio}"

    batch_size = inputs.size(0)
    sequence_length = inputs.size(-1)

    time_steps = int(time_ratio * sequence_length)

    if inputs.dim() == 3:
        features = inputs.size(1)
    else:
        features = 1

    # if features > sequence_length:
    #     print(f"Warning: features={features} is greater than sequence_length={sequence_length}")

    # Linear interpolation of weights from 'weight' to 1 over N steps
    global cached_weights

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
    average_weight = total_weight / sequence_length

    loss = torch.sum(weighted_loss)

    loss /= (batch_size * average_weight * sequence_length * features)

    return percentage * torch.clamp(loss, 0)



def reconstruction_loss(inputs, outputs):
    assert inputs.shape == outputs.shape, f"reconstruction_loss: shapes don't match, inputs={inputs.shape}, outputs={outputs.shape}"
    return basic_reconstruction_loss(inputs, outputs)

    # this does work too.
    return weighted_time_reconstruction_loss(inputs, outputs, weight=10)

# Test the basic loss & weighted loss:
if __name__ == '__main__':
    inputs = torch.randn(7, 100, 1000)
    outputs = inputs + (2 * torch.randn(inputs.shape) - 1) * 0.4
    time_steps = 1/3
    basic = basic_reconstruction_loss(inputs, outputs).item()
    loss1 = weighted_time_reconstruction_loss(inputs, outputs, 1, time_steps).item()
    print(f"basic={basic:.2f}, loss1={loss1:.2f}")
    assert abs(basic - loss1) < 1e-4

    lossW = weighted_time_reconstruction_loss(inputs, outputs, 10, time_steps).item()
    delta = basic/lossW - 1
    print(f"weighted loss={lossW:.2f}, vs basic={basic:.2f}, difference={100*delta:.2f}%")
    assert abs(delta) < 0.05 # we expect these two to be commensurate


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def _vae_loss_function(inputs, outputs, mu, logvar):

    distance  = reconstruction_loss(inputs, outputs)
    assert distance >= 0, f"negative distance={distance}"

    kl_div = kl_divergence(mu, logvar) / inputs.size(0)
    assert kl_div >= 0, f"negative kl_div={kl_div}"

    loss = distance + kl_div

    if np.random.random() < 1e-2:
        print(f"vae_loss={loss:.2f}, distance={distance:.2f}, kl_div={kl_div:.2f}")

    # We seem to get into situations where the reconstruction loss and the KL loss are fighting each other :(
    # Try to focus the optimiser on whichever loss is largest:
    #loss += distance * kl_div # sames as  (1 + distance) * (1 + kl_div) - 1

    if loss < 0:
        print(f"Negative loss!! loss={loss} (reconstruction={distance}, kl_divergence={kl_div}) in vae_loss_function")
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
        self.enable_variational(True)

    def enable_variational(self, enable=True):
        self.is_variational = enable
        if not self.is_variational:
            print("\n*** VAE is DISABLED! ***\n")

    def encode(self, x):
        assert x[0].shape == self.input_shape, f"VAE.encode expected shape={self.input_shape} but got {x[0].shape}"
        x = x.view(x.size(0), -1) # flatten

        if len(self.encoder_layers):
            x = self.encoder_layers(x)
            
        mu = self.fc_mu(x)

        if self.is_variational:
            logvar = self.fc_logvar(x)
        else:
            mu = nn.Tanh()(mu)
            logvar = torch.zeros(mu.shape).to(device)
        
        return mu, logvar


    def decode(self, z):
        x = self.decoder_layers(z)
        assert x.size(1) == total_elements(self.input_shape), f"VAE.decode expected {total_elements(self.input_shape)} outputs, but got {x.size(1)}"

        x = x.view((x.size(0), *self.input_shape)) # re-inflate
        #debug("decode.x", x)

        return x


    def forward(self, x):
        mu, logvar = self.encode(x)

        if self.is_variational:
            z = vae_reparameterize(mu, logvar)
        else:
            z = mu

        return self.decode(z), mu, logvar

    # For compatibility with the combined VAE
    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)

        loss = self.loss_function(inputs, outputs, mus, logvars)

        return loss, outputs

    def loss_function(self, inputs, outputs, mus, logvars):
        if self.is_variational:
            return _vae_loss_function(inputs, outputs, mus, logvars)
        else:
            return reconstruction_loss(inputs, outputs)


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
        loss = self.vae.loss_function(inputs, outputs, mus, logvars)
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
