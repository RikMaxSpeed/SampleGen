import torch.nn.functional as F
import copy

from ModelUtils import *

percentage = 100

# Loss functions
def basic_reconstruction_loss(inputs, outputs):
    return percentage * F.mse_loss(inputs, outputs)


cached_weights = {}

# Weight the samples over time: it's critical to get start of the sound correct.
def weighted_time_reconstruction_loss(inputs, outputs, weight, power, verbose=False):
    assert inputs.dim() >= 2, f"Expected inputs to be greater than 2, not {inputs.dim()}"
    assert weight >= 1, f"Expected weight to be greater than 1, not {weight}"
    assert power >= 1, f"Expected power={power} to be >= 1"

    batch_size = inputs.size(0)
    sequence_length = inputs.size(-1)

    time_steps = min(int(power * sequence_length), 1)
    assert time_steps < sequence_length, f"Expected time steps to be less than {sequence_length}, not {time_steps}"

    if inputs.dim() == 3:
        features = inputs.size(1)
    else:
        features = 1

    # if features > sequence_length: # warning heuristic, fails under extreme time compression
    #     print(f"Warning: features={features} is greater than sequence_length={sequence_length}")

    # Linear interpolation of weights from 'weight' to 1 over N steps
    global cached_weights

    cached_key = (sequence_length, time_steps, weight)
    weights = cached_weights.get(cached_key)
    if weights is None:
        t = torch.linspace(0, 1, sequence_length, device=inputs.device)
        weights = 1 + (weight - 1) * (torch.abs(t - 0.5) / 0.5) ** power
        weights /= weights.sum()
        print(f"weights={weights}")

        cached_weights[cached_key] = weights
        if verbose:
            print(f"weight={weight}, length={sequence_length}, time_steps={time_steps}, weights={weights}")

    loss = F.mse_loss(inputs, outputs, reduction='none')

    if inputs.dim() == 3:
        loss = torch.sum(loss, dim=1)

    loss = torch.sum(loss * weights)
    loss /= (batch_size * features) # normalise

    return percentage * torch.clamp(loss, 0)



def reconstruction_loss(inputs, outputs):
    assert inputs.shape == outputs.shape, f"reconstruction_loss: shapes don't match, inputs={inputs.shape}, outputs={outputs.shape}"
    #return basic_reconstruction_loss(inputs, outputs)

    # this does work too.
    return weighted_time_reconstruction_loss(inputs, outputs, weight=10, power=8)

# Test the basic loss & weighted loss:
if __name__ == '__main__':
    inputs = torch.randn(7, 100, 1000)
    outputs = inputs + (2 * torch.randn(inputs.shape) - 1) * 0.4
    basic = basic_reconstruction_loss(inputs, outputs).item()
    loss1 = weighted_time_reconstruction_loss(inputs, outputs, 1, 1).item()
    print(f"basic={basic:.2f}, loss1={loss1:.2f}")
    assert abs(basic - loss1) < 1e-4

    lossW = weighted_time_reconstruction_loss(inputs, outputs, 10, 3).item()
    delta = basic/lossW - 1
    print(f"weighted loss={lossW:.2f}, vs basic={basic:.2f}, difference={100*delta:.2f}%")
    assert abs(delta) < 0.05 # we expect these two to be commensurate


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# KL annealing: we'll evolve the KL divergence weight over time from 0 (target the reconstruction loss) to 1 (normalise the latent space variables)
kl_weight = 1
last_displayed_kl_weight = 0
def set_kl_weight(kl, epoch):
    global kl_weight, last_displayed_kl_weight

    if kl_weight != kl:
        kl_weight = kl
        if np.abs(last_displayed_kl_weight - kl) > 0.05:
            print(f"Using VAE KL divergence weight={100*kl_weight:.1f}% at epoch {epoch}")
            last_displayed_kl_weight = kl

def vae_loss_function(inputs, outputs, mu, logvar):

    distance  = reconstruction_loss(inputs, outputs)
    assert distance >= 0, f"negative distance={distance}"

    kl_div = kl_divergence(mu, logvar) / inputs.size(0)
    if kl_div < 0:
        print(f"negative kl_div={kl_div}")
        kl_div = 0

    loss = distance + kl_weight * (kl_div - distance) # linear interpolation between the 2 errors

    if loss < 0: # Usually a bad sign that the prior has collapsed.
        print(f"Negative loss!! loss={loss} (reconstruction={distance}, kl_divergence={kl_div}) in vae_loss_function")
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
            logvar = torch.full(mu.shape, -9999).to(get_device())
        
        return mu, logvar


    def decode(self, z):
        x = self.decoder_layers(z)
        assert x.size(1) == total_elements(self.input_shape), f"VAE.decode expected {total_elements(self.input_shape)} outputs, but got {x.size(1)}"

        x = x.view((x.size(0), *self.input_shape)) # re-inflate

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
            return vae_loss_function(inputs, outputs, mus, logvars)
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
