import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils import interpolate_layer_sizes, fully_connected_size
from VariationalAutoEncoder import *

# Here we create 2 models: the first applies an MLP to each spectrogram frame, the second combines this with the VAE.
# The first model can be trained and optimised independently. This is useful to prove that the model can work, and then determine its optimal configuration.
# In theory, the 2nd model could load trained version of the first model, and then only train the VAE encoding - but I haven't implemented this.


##########################################################################################
# Step-Wise MLP Auto-encoder (with no VAE)
# At each step in the STFT, we run a MLP using as input the previous and current frames, and output a 'control' result.
# The goals are that the MLP learns how to predict a frame from the previous frame, and to generate key descriptive parameters at each step.
# On further reading, it looks like this could be replaced with an RNN, I will give that a go!

class StepWiseMLPAutoEncoder(nn.Module):
        
    @staticmethod
    def get_layer_sizes(stft_buckets, hidden_size, depth, ratio):
        encode_layer_sizes = interpolate_layer_sizes(2 * stft_buckets, hidden_size, depth, ratio)
        decode_layer_sizes = interpolate_layer_sizes(stft_buckets + hidden_size, stft_buckets, depth, ratio)
        return encode_layer_sizes, decode_layer_sizes
        
    @staticmethod
    def approx_trainable_parameters(stft_buckets, hidden_size, depth, ratio):
        encode_layer_sizes, decode_layer_sizes = StepWiseMLPAutoEncoder.get_layer_sizes(stft_buckets, hidden_size, depth, ratio)
        encode_size = fully_connected_size(encode_layer_sizes)
        decode_size = fully_connected_size(decode_layer_sizes)
        total = encode_size + decode_size
        return total


    def __init__(self, stft_buckets, sequence_length, hidden_size, depth, ratio):
        super(StepWiseMLPAutoEncoder, self).__init__()
        
        self.stft_buckets = stft_buckets
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        encode_layer_sizes, decode_layer_sizes = StepWiseMLPAutoEncoder.get_layer_sizes(stft_buckets, hidden_size, depth, ratio)
        
        self.encoder = build_sequential_model(encode_layer_sizes, None)
        self.decoder = build_sequential_model(decode_layer_sizes, None) #was nn.Tanh())

        print(f"StepWiseMLPAutoEncoder {count_trainable_parameters(self):,} parameters, compression={stft_buckets/hidden_size:.1f}")


    def encode(self, x):
        batch_size = x.size(0)
        assert(x.size(1) == self.stft_buckets)
        assert(x.size(2) == self.sequence_length)
        
        controls = torch.zeros(batch_size, self.hidden_size, self.sequence_length).to(device)
        prev_stft = torch.zeros(batch_size, self.stft_buckets).to(device)
        
        # Process each time step
        for t in range(self.sequence_length):
            # Concatenate previous STFT, current STFT
            curr_stft = x[:, :, t]
            combined_input = torch.cat([prev_stft, curr_stft], dim=1)
            
            # Update control parameters
            result = self.encoder(combined_input)
            controls[:, :, t] = result

            # Update previous STFT frame
            prev_stft = curr_stft #.clone() do we need to clone here??


        controls = controls.flatten(-2)
        assert(controls.size(0) == batch_size)
        assert(controls.size(1) == self.sequence_length * self.hidden_size)
        
        return controls


    def decode(self, x):
        batch_size = x.size(0)
        assert(x.size(1) == self.sequence_length * self.hidden_size)
        
        controls = x.view(batch_size, self.hidden_size, self.sequence_length).to(device)
        prev_stft = torch.zeros(batch_size, self.stft_buckets).to(device)
        reconstructed = torch.zeros(batch_size, self.stft_buckets, self.sequence_length).to(device)

        # Process each time step
        for t in range(self.sequence_length):

            # Concatenate control parameters and previous STFT
            combined_input = torch.cat([controls[:, :, t], prev_stft], dim=1)
            
            # Update previous STFT frame with the output of the decoder
            prev_stft = self.decoder(combined_input)

            # Store the reconstructed STFT frame
            reconstructed[:, :, t] = prev_stft # do we need .clone() here?

        return reconstructed


    def forward(self, inputs):
        controls = self.encode(inputs)
        outputs = self.decode(controls)
        return outputs
        
        
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs




##########################################################################################
# Step-Wise MLP Auto-encoder with VAE
# Here we combine the StepWiseMLP model with the VAE auto-encoder.
# It might actually be possible to train them separately, ie: first optimise the StepWiseMLP, then use the VAE to further compress the data.

class StepWiseMLP_VAE(nn.Module):
    @staticmethod
    def get_vae_layers(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        stepwise_output_size = hidden_size * sequence_length
        layers = interpolate_layer_sizes(stepwise_output_size, latent_size, vae_depth, vae_ratio)
        return layers


    @staticmethod
    def approx_trainable_parameters(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        stepwise = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, hidden_size, depth, ratio)
        vae_layers = StepWiseMLP_VAE.get_vae_layers(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        vae = VariationalAutoEncoder.approx_trainable_parameters(vae_layers)
        return stepwise + vae


    def __init__(self, stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        super(StepWiseMLP_VAE, self).__init__()
        
        self.stepwise = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, hidden_size, depth, ratio)
        
        vae_layers = StepWiseMLP_VAE.get_vae_layers(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        self.vae = VariationalAutoEncoder(vae_layers)

    def encode(self, x):
        controls = self.stepwise.encode(x)
        mu, logvar = self.vae.encode(controls)
        return mu, logvar


    def decode(self, z):
        controls = self.vae.decode(z)
        stft = self.stepwise.decode(controls)
        return stft


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = vae_reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
        
    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = vae_loss_function(inputs, outputs, mus, logvars)
        return loss, outputs


    def load_outer_layers(self, file_name):
        self.stepwise.load_outer_layers(file_name)

    def freeze_outer_layers(self):
        self.stepwise.freeze_outer_layers()
