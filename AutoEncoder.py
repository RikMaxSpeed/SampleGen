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



# Basic auto-encoder with no VAE

class HybridCNNAutoEncoder(nn.Module):
    @staticmethod
    def estimate_parameters(stft_buckets, seq_length, num_conv_filters, rnn_hidden_size, input_size):
        # Parameters in the 1D Conv layer
        conv_params = (stft_buckets * num_conv_filters * 3) + num_conv_filters  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        # Parameters in the GRU layer - Encoder
        # GRU parameters = 3 * (hidden_size^2 + hidden_size * input_size + hidden_size) * num_layers
        rnn_input_size = num_conv_filters * input_size
        encoder_rnn_params = 3 * (rnn_hidden_size**2 + rnn_hidden_size * rnn_input_size + rnn_hidden_size)

        # Parameters in the GRU layer - Decoder (similar to encoder)
        decoder_rnn_params = encoder_rnn_params #3 * (rnn_hidden_size**2 + rnn_hidden_size * rnn_input_size + rnn_hidden_size)

        # Parameters in the 1D Transposed Conv layer
        deconv_params = (num_conv_filters * stft_buckets * 3) + stft_buckets  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        total_params = conv_params + encoder_rnn_params + decoder_rnn_params + deconv_params
        return total_params


    def __init__(self, stft_buckets, seq_length, num_conv_filters, rnn_hidden_size, input_size):
        super(HybridCNNAutoEncoder, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.rnn_hidden_size = rnn_hidden_size

        # Encoder
        self.encoder_conv1d = nn.Conv1d(in_channels=stft_buckets, out_channels=num_conv_filters, kernel_size=3, stride=1, padding=1)
        self.encoder_relu = nn.ReLU()
        self.encoder_rnn = nn.GRU(input_size=num_conv_filters * input_size, hidden_size=rnn_hidden_size, batch_first=True)

        # Decoder
        self.decoder_rnn = nn.GRU(input_size=rnn_hidden_size, hidden_size=num_conv_filters * input_size, batch_first=True)
        self.decoder_conv1d = nn.ConvTranspose1d(in_channels=num_conv_filters, out_channels=stft_buckets, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = self.encoder_conv1d(x)
        x = self.encoder_relu(x)
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
        


