import torch
import torch.nn as nn
import torch.nn.functional as F
from Debug import *
from MakeSTFTs import * # required for some global variables :(
from ModelUtils import *


# Loss functions
def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')
        
        
def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


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
        
        
    def loss_function(self, recon_x, x, mu, logvar):
        error  = reconstruction_loss(recon_x, x)
        kl_div = kl_divergence(mu, logvar)

        #print("error={:.3f}, kl_divergence={:.3f}, ratio={:.1f}".format(error, kl_div, kl_div/error))
        
        # The optimiser appears to be able to efficiently minimise the KL loss, so it's unneccesary to weight it vs the reconstruction loss.
        kl_weight = 1.0
        
        return (error + kl_weight * kl_div) / x[0].numel()



class STFTVariationalAutoEncoder(nn.Module):
    @staticmethod
    def estimate_parameters(sequence_length, stft_buckets, sizes):
        sizes = [sequence_length * stft_buckets] + sizes
        return fully_connected_size(sizes) * 2 # encode + decode
        # Note: this approximate but good enough in practice


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


    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs, True)
        loss = self.loss_function(outputs, inputs, mus, logvars)
        return loss



##########################################################################################
# CNN/RNN Auto-encoder with no VAE
#

class HybridCNNAutoEncoder(nn.Module):
    @staticmethod
    def estimate_parameters(stft_buckets, seq_length, kernel_count, kernel_size, rnn_hidden_size):
        # Parameters in the 1D Conv layer
        conv_params = (stft_buckets * kernel_count * kernel_count) + kernel_count  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        # Parameters in the GRU layer - Encoder
        rnn_input_size = kernel_count * seq_length
        encoder_rnn_params = 3 * (rnn_hidden_size**2 + rnn_hidden_size * seq_length + rnn_hidden_size)

        # Parameters in the GRU layer - Decoder (similar to encoder)
        decoder_rnn_params = encoder_rnn_params #3 * (rnn_hidden_size**2 + rnn_hidden_size * rnn_input_size + rnn_hidden_size)

        # Parameters in the 1D Transposed Conv layer
        deconv_params = (kernel_count * stft_buckets * kernel_count) + stft_buckets  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        total_params = conv_params + encoder_rnn_params + decoder_rnn_params + deconv_params
        return total_params


    def __init__(self, stft_buckets, seq_length, kernel_count, kernel_size, rnn_hidden_size):
        super(HybridCNNAutoEncoder, self).__init__()
        self.seq_length = seq_length
        self.rnn_hidden_size = rnn_hidden_size
        self.encoded_size = rnn_hidden_size * seq_length
        print(f"sequence_length={seq_length}, encoded_size={self.encoded_size}")
        # Encoder
        self.encoder_conv1d = nn.Conv1d(in_channels=stft_buckets, out_channels=kernel_count, kernel_size=kernel_size, stride=1, padding=1)
        print(f"encoder_conv1d={self.encoder_conv1d}")
    
        self.encoder_rnn = nn.GRU(input_size=kernel_count * seq_length, hidden_size=rnn_hidden_size, batch_first=True)
        print(f"encoder_rnn={self.encoder_rnn}")

        # Decoder
        self.decoder_rnn = nn.GRU(input_size=kernel_count * input_size, hidden_size=rnn_hidden_size, batch_first=True)
        print(f"decoder_rnn={self.decoder_rnn}")
        
        self.decoder_conv1d = nn.ConvTranspose1d(in_channels=kernel_count, out_channels=stft_buckets, kernel_size=kernel_size, stride=1, padding=1)
        print(f"decoder_conv1d={self.decoder_conv1d}")
        
        
    def encode(self, x):
        debug("encode.x", x)
        x = self.encoder_conv1d(x)
        debug("encoder_conv1d.x", x)
        x = F.relu(x)
        debug("relu.x", x)
        x = x.view(x.size(0), self.seq_length, -1)
        x, _ = self.encoder_rnn(x)
        return x.flatten()


    def decode(self, x):
        x = x.view(x.shape[0], self.seq_length, self.rnn_hidden_size)
        x, _ = self.decoder_rnn(x)
        x = x.view(x.size(0), -1, self.input_size)
        x = self.decoder_conv1d(x)
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
        
        
    def forward_loss(self, inputs):
        x = self.forward(inputs)
        return reconstruction_loss(x, inputs)


##########################################################################################
# Top-Level to create models and read hyper-parameters
#

model_type = None

def set_model_type(name):
    global model_type
    if model_type != name:
        model_type = name
        print(f"Using model={model_type}")


def get_layers(model_params):
    latent_size, layer3_ratio, layer2_ratio, layer1_ratio = model_params

    # Translate the ratios into actual sizes: this ensures we have increasing layer sizes
    layer3_size = int(latent_size * layer3_ratio)
    layer2_size = int(layer3_size * layer2_ratio)
    layer1_size = int(layer2_size * layer1_ratio)
    assert(latent_size <= layer3_size <= layer2_size <= layer1_size)
    
    layers = [stft_buckets * sequence_length, layer1_size, layer2_size, layer3_size, latent_size]

    return layers
    
    




def make_model(model_params, max_params, verbose):
    invalid_model = None, None
    
    if model_type == "VAE_MLP":
        latent_size, layer3_ratio, layer2_ratio, layer1_ratio = model_params
        layers = get_layers(model_params)
        approx_size = 2 * fully_connected_size(layers)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type}: latent={layers[4]}, layer3={layers[3]}, layer2={layers[2]}, layer1={layers[1]}"
        model = STFTVariationalAutoEncoder(sequence_length, stft_buckets, layers[1:], nn.ReLU())

    elif model_type == "Hybrid_CNN":
        kernel_count, kernel_size, rnn_hidden_size = model_params
        
        # for some reason we get int64 here which upsets PyTorch...
        kernel_count    = int(kernel_count)
        kernel_size     = int(kernel_size)
        rnn_hidden_size = int(rnn_hidden_size)
        
        approx_size = HybridCNNAutoEncoder.estimate_parameters(stft_buckets, sequence_length, kernel_count, kernel_size, rnn_hidden_size)
        print(f"approx_size={approx_size:,} parameters")
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type}: kernels={kernel_count}, kernel_size={kernel_size}, rnn_hidden={rnn_hidden_size}"
        print(model_text)
        model = HybridCNNAutoEncoder(stft_buckets, sequence_length, kernel_count, kernel_size, rnn_hidden_size)

    else:
        raise Exception(f"Unknown model: {model_type}")
        
    
    # Check the real size:
    size = count_trainable_parameters(model)
    print(f"model={model_type}, approx size={approx_size:,} parmaters, exact={size:,}, error={100*(approx_size/size - 1):.2f}%")
    
    if size > max_params:
        print(f"Model is too large: {size:,} parameters vs max={max_params:,}")
        return invalid_model

    # Get ready!
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    if verbose:
        print("model={}".format(model))
    
    return model, model_text

