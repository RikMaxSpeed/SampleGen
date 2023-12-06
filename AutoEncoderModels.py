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
            
        model_text = f"{model_type} latent={layers[4]}, layer3={layers[3]}, layer2={layers[2]}, layer1={layers[1]}"
        model = STFTVariationalAutoEncoder(sequence_length, stft_buckets, layers[1:], nn.ReLU())
        
    elif model_type == "StepWiseMLP":
        control_size, depth, ratio = model_params
        approx_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, control_size, depth, ratio)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type} control={control_size}, depth={depth}, ratio={ratio:.2f}"
        model = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio)
            
    elif model_type == "StepWiseVAEMLP":
        control_size, depth, ratio, latent_size, vae_depth, vae_ratio = model_params
        approx_size = StepWiseVAEMLPAutoEncoder.approx_trainable_parameters(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type} control={control_size}, depth={depth}, ratio={ratio:.2f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
        model = StepWiseVAEMLPAutoEncoder(stft_buckets, sequence_length, control_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    
    elif model == "RNNAutoEncoder":
        hidden_size, encode_depth, decode_depth = model_params
        dropout = 0 # will explore this later.
        approx_size = StepWiseVAEMLPAutoEncoder.approx_trainable_parameters(stft_buckets, hidden_size, encode_depth, decode_depth)
        if approx_size > max_params:
            print(f"Model is too large: approx {size:,} parameters vs max={max_params:,}")
            return invalid_model
            
        model_text = f"{model_type} hidden={hidden_size}, encode_depth={encode_depth}, decode_depth={decode_depth}"
        model = RNNAutoEncoder(stft_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout)
    
            
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
            
        model_text = f"{model_type} kernels={kernel_count}, kernel_size={kernel_size}, rnn_hidden={rnn_hidden_size}"
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

