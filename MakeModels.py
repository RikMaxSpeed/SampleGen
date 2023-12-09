from Debug import *
from MakeSTFTs import *
from ModelUtils import *
from STFT_VAE import *
from MLP_VAE import *
from RNN_VAE import *


##########################################################################################
# Top-Level to create models and read hyper-parameters
#

model_type = None

def set_model_type(name):
    global model_type
    if model_type != name:
        model_type = name
        print(f"Using model={model_type}")


def is_too_large(approx_size, max_params):
    if approx_size > max_params:
            print(f"Model is too large: approx {approx_size:,} parameters vs max={max_params:,}")
            return True
    else:
        return False

def make_model(model_params, max_params, verbose):
    invalid_model = None, None
    
    if model_type == "STFT_VAE":
        latent_size, depth, ratio = model_params
        model_text = f"{model_type} latent={latent_size}, layers={depth}, ratio={ratio:.2f}"
        print(model_text)
        approx_size = STFTVariationalAutoEncoder.approx_trainable_parameters(stft_buckets, sequence_length, latent_size, depth, ratio)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = STFTVariationalAutoEncoder(stft_buckets, sequence_length, latent_size, depth, ratio)
        
    elif model_type == "StepWiseMLP":
        control_size, depth, ratio = model_params
        model_text = f"{model_type} control={control_size}, depth={depth}, ratio={ratio:.2f}"
        print(model_text)
        approx_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, control_size, depth, ratio)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio)
            
    elif model_type == "StepWiseVAEMLP":
        control_size, depth, ratio, latent_size, vae_depth, vae_ratio = model_params
        model_text = f"{model_type} control={control_size}, depth={depth}, ratio={ratio:.2f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
        print(model_text)
        approx_size = StepWiseMLP_VAE.approx_trainable_parameters(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = StepWiseMLP_VAE(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    
    elif model_type == "RNNAutoEncoder":
        hidden_size, encode_depth, decode_depth = model_params
        model_text = f"{model_type} hidden={hidden_size}, encode_depth={encode_depth}, decode_depth={decode_depth}"
        print(model_text)
        dropout = 0 # will explore this later.
        approx_size = RNNAutoEncoder.approx_trainable_parameters(stft_buckets, hidden_size, encode_depth, decode_depth)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = RNNAutoEncoder(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout)
    
    elif model_type == "RNN_VAE":
        hidden_size, encode_depth, decode_depth, latent_size, vae_depth, vae_ratio = model_params
        model_text = f"{model_type} hidden={hidden_size}, encode_depth={encode_depth}, decode_depth={decode_depth}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
        print(model_text)
        dropout = 0 # will explore this later.
        approx_size = RNN_VAE.approx_trainable_parameters(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, latent_size, vae_depth, vae_ratio)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = RNN_VAE(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout, latent_size, vae_depth, vae_ratio)

    else:
        raise Exception(f"Unknown model: {model_type}")


    # Check the real size:
    size = count_trainable_parameters(model)
    print(f"model={model_type}, approx size={approx_size:,} parameters, exact={size:,}, difference={100*(approx_size / size - 1):.4f}%")
    model_text += f" (size={size:,})"
    
    if size > max_params:
        print(f"Model is too large: {size:,} parameters vs max={max_params:,}")
        return invalid_model

    # Get ready!
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    if verbose:
        print("model={}".format(model))
    
    return model, model_text
