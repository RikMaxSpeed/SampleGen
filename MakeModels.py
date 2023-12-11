from Debug import *
from MakeSTFTs import *
from ModelUtils import *
from STFT_VAE import *
from MLP_VAE import *
from RNN_VAE import *



def make_stepwiseMLPVAE(params, max_params):
    print(f"make_stepwiseMLPVAE: {params}")
    
    hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio = params
    model_text = f"{model_type} control={hidden_size}, depth={depth}, ratio={ratio:.2f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
    print(model_text)
    
    approx_size = StepWiseMLP_VAE.approx_trainable_parameters(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    if is_too_large(approx_size, max_params):
        return None, model_text
        
    model = StepWiseMLP_VAE(stft_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    return model, model_text


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
        hidden_size, depth, ratio = model_params
        model_text = f"{model_type} control={hidden_size}, depth={depth}, ratio={ratio:.2f}"
        print(model_text)
        approx_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(stft_buckets, hidden_size, depth, ratio)
        if is_too_large(approx_size, max_params):
            return invalid_model
            
        model = StepWiseMLPAutoEncoder(stft_buckets, sequence_length, hidden_size, depth, ratio)
            
    elif model_type == "StepWiseVAEMLP":
        model, model_text = make_stepwiseMLPVAE(model_params, max_params)
    
    elif model_type == "Incremental_StepWiseVAEMLP":
        # Load the pre-trained model
        mlp_params, file_name = get_best_configuration_for_model("StepWiseMLP")
        new_params = mlp_params[3:6] + model_params

        model, model_text = make_stepwiseMLPVAE(new_params, max_params)
        if model is None:
            return invalid_model
        
        # Incremental training: load the previous saved state, and freeze the layers we won't re-train
        model.load_outer_layers(file_name)
        model.freeze_outer_layers()
    
    
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
    print(f"{model_type} {size:,} parameters")
    model_text += f" ({size:,} parameters)"
    
    # Warn if the approximation was off:
    size_error = approx_size / size  - 1
    if np.abs(size_error) > 0.01:
        print("Innaccurate approximate size={approx_size:,} vs actual size={size:,}, error={100*size_error:.2f}%")
    
    # Too big?
    if size > max_params:
        print(f"Model is too large: {size:,} parameters vs max={max_params:,}")
        return invalid_model, model_text

    # Get ready!
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    if verbose:
        print("model={}".format(model))
    
    return model, model_text



# Dictionary of best models we've built
best_models = {
"StepWiseMLP": ([4, 2.214254104518763e-05, 5.055883450850856e-08, 165, 4, 0.3187354618451725],
                "StepWiseMLP control=165, depth=4, ratio=0.32 (5,292,057 parameters).wab"), # Small model with loss=85

#*** Best! loss=791.04
#"StepWiseVAEMLP": ([64, 1e-5, 0.001, 43, 5, 0.13753954871555363, 8, 2, 0.18245971697542837], "No file"),
"StepWiseVAEMLP": ([16, 1.4815996677501001e-05, 0.0016767313292796594, 43, 5, 0.13753954871555363, 8, 2, 0.18245971697542837], "No file"),

    "RNN_VAE": ([18, 0.0005575544181212729, 5.294016993959888e-06, 29, 1, 2, 4, 4, 0.20679719844604053],
                "StepWiseVAEMLP control=48, depth=2, ratio=0.50, latent=6, VAE depth=4, VAE ratio=1.43.wab"), # train loss=0.01244, test  loss=0.01417
}

def get_best_configuration_for_model(model):
    set_model_type(model)
    return best_models[model]

def get_best_model_configuration():
    #best = "RNN_VAE"
    best = "StepWiseMLP"
    return get_best_configuration_for_model(best)


def load_best_model():
    params, file_name = get_best_model_configuration()
    model_params = params[3:]
    max_params = +1e99 # ignore
    verbose = True
    model, model_text = make_model(model_params, max_params, verbose)
    
    print(f"Loading weights & biases from file '{file_name}'")
    model.load_state_dict(torch.load(file_name))
    model.eval() # Ensure the model is in evaluation mode
    model.to(device)
    print(f"{model.__class__.__name__} has {count_trainable_parameters(model):,} weights & biases")
    
    return model
