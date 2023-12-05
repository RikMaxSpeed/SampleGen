import torch
import torch.nn as nn
import torch.nn.functional as F
from Debug import *
from MakeSTFTs import * # required for some global variables :(
from ModelUtils import *


# Loss functions
def reconstruction_loss(inputs, outputs):
    assert(inputs.shape == outputs.shape)
    return F.mse_loss(inputs, outputs, reduction='sum')


def kl_divergence(mu, logvar):
    # see https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Generic VAE composed of a number of fully connected linear layers (re-usable)
class VariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(sizes):
        return fully_connected_size(sizes) * 2 # encode + decode
        # Note: approximate but good enough in practice
        
        
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

        print(f"VariationalAutoEncoder compression: {sizes[0]/sizes[-1]:.1f} x smaller")


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
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        error  = reconstruction_loss(inputs, outputs)
        kl_div = kl_divergence(mu, logvar)

        #print("error={:.3f}, kl_divergence={:.3f}, ratio={:.1f}".format(error, kl_div, kl_div/error))
        
        # The optimiser appears to be able to efficiently minimise the KL loss, so it's unneccesary to weight it vs the reconstruction loss.
        kl_weight = 1.0
        
        return (error + kl_weight * kl_div) / inputs[0].numel()


class STFTVariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(sequence_length, stft_buckets, sizes):
        sizes = [sequence_length * stft_buckets] + sizes
        return VariationalAutoEncoder.approx_trainable_parameters(sizes)


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
        
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x, mu, logvar = self.vae.forward(x)
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x, mu, logvar


    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)


    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs, True)
        loss = self.loss_function(inputs, outputs, mus, logvars)
        return loss, outputs


##########################################################################################
# Step-Wise MLP Auto-encoder with no VAE
# At each step in the STFT, we run a MLP using as input the previous and current frames, and output a 'control' result.
# The goals are that the MLP learns how to predict a frame from the previous frame, and to generate key descriptive parameters at each step.
# On further reading, it looks like this could be replaced with an RNN, I will give that a go!

class StepWiseMLPAutoEncoder(nn.Module):
        
    @staticmethod
    def get_layer_sizes(stft_buckets, control_size, depth, ratio):
        encode_layer_sizes = interpolate_layer_sizes(1 + 2 * stft_buckets, control_size, depth, ratio)
        decode_layer_sizes = interpolate_layer_sizes(1 + stft_buckets + control_size, stft_buckets, depth, ratio)
        return encode_layer_sizes, decode_layer_sizes
        
    @staticmethod
    def approx_trainable_parameters(stft_buckets, control_size, depth, ratio):
        encode_layer_sizes, decode_layer_sizes = StepWiseMLPAutoEncoder.get_layer_sizes(stft_buckets, control_size, depth, ratio)
        encode_size = fully_connected_size(encode_layer_sizes)
        decode_size = fully_connected_size(decode_layer_sizes)
        total = encode_size + decode_size
        print(f"encode={encode_layer_sizes}={encode_size:,}, decode={decode_layer_sizes}={decode_size:,}, total={total:,}")
        return total

    def __init__(self, stft_buckets, sequence_length, control_size, depth, ratio):
        super(StepWiseMLPAutoEncoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.control_size = control_size
        print(f"StepWiseMLPAutoEncoder compression: {stft_buckets/control_size:.1f} x smaller")
        encode_layer_sizes, decode_layer_sizes = StepWiseMLPAutoEncoder.get_layer_sizes(stft_buckets, control_size, depth, ratio)
        
        self.encoder = build_sequential_model(encode_layer_sizes)
        self.decoder = build_sequential_model(decode_layer_sizes)

    def encode(self, x):
        batch_size = x.size(0)
        controls = torch.zeros(batch_size, self.sequence_length, self.control_size).to(device)
        prev_stft = torch.zeros(batch_size, stft_buckets).to(device)
        
        # Process each time step
        for t in range(self.sequence_length):
            t_tensor = torch.full((batch_size, 1), t / float(self.sequence_length), dtype=torch.float32).to(device)
            
            # Concatenate previous STFT, current STFT, and time step
            curr_stft = x[:, :, t]
            combined_input = torch.cat([prev_stft, curr_stft, t_tensor], dim=1)
            
            # Update control parameters
            result = self.encoder(combined_input)
            controls[:, t, :] = result

            # Update previous STFT frame
            prev_stft = curr_stft.clone()

        controls = controls.flatten(-2)
        return controls


    def decode(self, x):
        batch_size = x.size(0)
        controls = x.view(batch_size, self.sequence_length, self.control_size).to(device)
        prev_stft = torch.zeros(batch_size, stft_buckets).to(device)
        reconstructed = torch.zeros(batch_size, stft_buckets, self.sequence_length).to(device)

        # Process each time step
        for t in range(self.sequence_length):
            t_tensor = torch.full((batch_size, 1), t / float(self.sequence_length), dtype=torch.float32).to(device)

            # Concatenate control parameters and previous STFT
            combined_input = torch.cat([controls[:, t, :], prev_stft, t_tensor], dim=1)
            
            # Update previous STFT frame with the output of the decoder
            prev_stft = self.decoder(combined_input)

            # Store the reconstructed STFT frame
            reconstructed[:, :, t] = prev_stft

        return reconstructed


    def forward(self, inputs):
        controls = self.encode(inputs)
        outputs = self.decode(controls)
        return outputs
        
        
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs) / inputs[0].numel()
        return loss, outputs


##########################################################################################
# Step-Wise MLP Auto-encoder with VAE
# Here we combine the StepWiseMLP model with the VAE auto-encoder.
# It might actually be possible to train them separately, ie: first optimise the StepWiseMLP, then use the VAE to further compress the data.

class StepWiseVAEMLPAutoEncoder(nn.Module):
    @staticmethod
    def get_vae_layers(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        stepwise_output_size = control_size * sequence_length
        layers = interpolate_layer_sizes(stepwise_output_size, latent_size, vae_depth, vae_ratio)
        print(f"VAE_layers={layers}")
        return layers


    @staticmethod
    def approx_trainable_parameters(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        stepwise = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, control_size, depth, ratio)
        vae_layers = StepWiseVAEMLPAutoEncoder.get_vae_layers(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        vae = VariationalAutoEncoder.approx_trainable_parameters(vae_layers)
        return stepwise + vae


    def __init__(self, stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio):
        super(StepWiseVAEMLPAutoEncoder, self).__init__()
        
        self.stepwise = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio)
        
        display(self.stepwise)
        
        vae_layers = StepWiseVAEMLPAutoEncoder.get_vae_layers(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        self.vae = VariationalAutoEncoder(vae_layers)
        
        display(self.vae)


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
        z = self.vae.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
        
    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)
        
        
    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs)
        loss = self.loss_function(inputs, outputs, mus, logvars)
        return loss, outputs

    




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
        
    elif model_type == "StepWiseMLP":
        control_size, depth, ratio = model_params
        approx_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, control_size, depth, ratio)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type}: control={control_size}, depth={depth}, ratio={ratio:.2f}"
        model = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio)
            
    elif model_type == "StepWiseVAEMLP":
        control_size, depth, ratio, latent_size, vae_depth, vae_ratio = model_params
        approx_size = StepWiseVAEMLPAutoEncoder.approx_trainable_parameters(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type}: control={control_size}, depth={depth}, ratio={ratio:.2f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
        model = StepWiseVAEMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    
            
    elif model_type == "Hybrid_CNN": # This didn't work
        kernel_count, kernel_size, rnn_hidden_size = model_params
        
        # for some reason we get int64 here which upsets PyTorch...
        kernel_count    = int(kernel_count)
        kernel_size     = int(kernel_size)
        rnn_hidden_size = int(rnn_hidden_size)
        
        approx_size = HybridCNNAutoEncoder.approx_trainable_parameters(stft_buckets, sequence_length, kernel_count, kernel_size, rnn_hidden_size)
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
    print(f"model={model_type}, approx size={approx_size:,} parameters, exact={size:,}, error={100*(approx_size/size - 1):.2f}%")
    
    if size > max_params:
        print(f"Model is too large: {size:,} parameters vs max={max_params:,}")
        return invalid_model

    # Get ready!
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    if verbose:
        print("model={}".format(model))
    
    return model, model_text

