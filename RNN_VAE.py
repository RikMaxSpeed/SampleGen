from VariationalAutoEncoder import *

# Buildling on the experience of the rnnVAE, we now try using an RNN to extract the time hiddens of the signal.
# We then combine that with the VAE to massively reduce the feature space.

##########################################################################################

class RNNAutoEncoder(nn.Module): # no VAE
                
    @staticmethod
    def approx_trainable_parameters(stft_buckets, hidden_size, encode_depth, decode_depth):
        encode = rnn_size(stft_buckets, hidden_size, encode_depth)
        decode = rnn_size(hidden_size, stft_buckets, decode_depth)
        return encode + decode

    def __init__(self, stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout):
        super(RNNAutoEncoder, self).__init__()
        
        hidden_size = int(hidden_size) # RNN objects to int64.
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        print(f"RNNAutoEncoder compression: {stft_buckets/hidden_size:.1f} x smaller")
        
        # RNN: input = (batch, sequence, features)
        self.encoder = torch.nn.RNN(stft_buckets, hidden_size, num_layers = encode_depth, batch_first = True, dropout = dropout)
        self.decoder = torch.nn.RNN(hidden_size, stft_buckets, num_layers = decode_depth, batch_first = True, dropout = dropout)
        # Important: in the rnnVAE I gave the model the previous and the current frame at each time step + the time itself.
        # In this version I'm only giving the model the individual time-steps... I don't know whether this will work.
        
    def encode(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1) # Swap from [batch, stft, seq] to [batch, seq, stft] which is what the RNN expects
        hiddens, _ = self.encoder(x) # also returns the last internal state
        hiddens = hiddens.flatten(-2)
        return hiddens

    def decode(self, x):
        batch_size = x.size(0)
        hiddens = x.view(batch_size, self.sequence_length, self.hidden_size).to(device)
        reconstructed, _ = self.decoder(hiddens)
        reconstructed = reconstructed.transpose(2, 1)
        return reconstructed

    def forward(self, inputs):
        hiddens = self.encode(inputs)
        outputs = self.decode(hiddens)
        return outputs
                
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs) / inputs[0].numel()
        return loss, outputs


##########################################################################################
# Here we combine the RNN-AutoEncoder with the VAE auto-encoder.
# It might actually be possible to train them separately, ie: first optimise the RNN, then use the VAE to further compress the data.

class RNN_VAE(nn.Module):
    @staticmethod
    def get_vae_layers(stft_buckets, sequence_length, hidden_size, latent_size, vae_depth, vae_ratio):
        rnn_output_size = hidden_size * sequence_length
        layers = interpolate_layer_sizes(rnn_output_size, latent_size, vae_depth, vae_ratio)
        print(f"VAE_layers={layers}")
        return layers


    @staticmethod
    def approx_trainable_parameters(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, latent_size, vae_depth, vae_ratio):
        rnn = RNNAutoEncoder.approx_trainable_parameters(stft_buckets, hidden_size, encode_depth, decode_depth)
        vae_layers = RNN_VAE.get_vae_layers(stft_buckets, sequence_length, hidden_size, latent_size, vae_depth, vae_ratio)
        vae = VariationalAutoEncoder.approx_trainable_parameters(vae_layers)
        return rnn + vae


    def __init__(self, stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout, latent_size, vae_depth, vae_ratio):
        super(RNN_VAE, self).__init__()
        
        self.rnn = RNNAutoEncoder(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout)
        
        vae_layers = RNN_VAE.get_vae_layers(stft_buckets, sequence_length, hidden_size, latent_size, vae_depth, vae_ratio)
        self.vae = VariationalAutoEncoder(vae_layers)
        

    def encode(self, x):
        hiddens = self.rnn.encode(x)
        mu, logvar = self.vae.encode(hiddens)
        return mu, logvar


    def decode(self, z):
        hiddens = self.vae.decode(z)
        stft = self.rnn.decode(hiddens)
        return stft


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)
        
        
    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = self.loss_function(inputs, outputs, mus, logvars)
        return loss, outputs