import Conv2D_AE
from Debug import *
from MakeSTFTs import *
from ModelUtils import *
from STFT_VAE import *
from MLP_AE import *
from RNN_AE import *
from RNN_FaT import *
from Conv2D_AE import *
from AudioConv_AE import *

import ast



def make_stepwiseMLPVAE(params, max_params):

    hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio = params
    model_text = f"{model_type} control={hidden_size}, depth={depth}, ratio={ratio:.2f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
    print(model_text)
    
    approx_size = StepWiseMLP_VAE.approx_trainable_parameters(freq_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    if is_too_large(approx_size, max_params):
        return None, model_text, approx_size
        
    model = StepWiseMLP_VAE(freq_buckets, sequence_length, hidden_size, depth, ratio, latent_size, vae_depth, vae_ratio)
    
    return model, model_text, approx_size


def make_RNN_VAE(model_type, model_params, max_params):
    hidden_size, encode_depth, decode_depth, latent_size, vae_depth, vae_ratio = model_params
    model_text = f"{model_type} hidden={hidden_size}, encode_depth={encode_depth}, decode_depth={decode_depth}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
    print(model_text)

    vae_sizes = interpolate_layer_sizes(hidden_size * sequence_length, latent_size, vae_depth, vae_ratio)
    print(f"VAE layers={vae_sizes}")

    rnn_size = RNNAutoEncoder.approx_trainable_parameters(freq_buckets, hidden_size, encode_depth, decode_depth)
    vae_size = VariationalAutoEncoder.approx_trainable_parameters(vae_sizes)
    approx_size = rnn_size + vae_size
    print(f"RNN={rnn_size:,}, VAE={vae_size:,}, approx total={approx_size:,}")
    
    if is_too_large(approx_size, max_params):
        return None, model_text, approx_size, vae_size
    
    
    dropout = 0
    rnn = RNNAutoEncoder(freq_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout)
    
    model = CombinedVAE(rnn, vae_sizes)
    
    return model, model_text, approx_size, vae_size

min_compression =  13 # or the VAE won't work
max_compression = 100 # or the auto-encoder won't work

def make_Conv2D_VAE(model_type, model_params, max_params):
    layer_count, kernel_count, kernel_size, latent_size, vae_depth, vae_ratio = model_params
    model_text = f"{model_type} conv layers={layer_count}, kernels={kernel_count}, size={kernel_size}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
    print(model_text)

    conv2d = Conv2DAutoEncoder(freq_buckets, sequence_length, layer_count, kernel_count, kernel_size)

    # we'd need to know the output size of the CNN...
    conv2_hidden = conv2d.encoded_size
    vae_sizes = interpolate_layer_sizes(conv2_hidden, latent_size, vae_depth, vae_ratio)
    print(f"VAE layers={vae_sizes}")

    conv2d_size = Conv2DAutoEncoder.approx_trainable_parameters(layer_count, kernel_count, kernel_size)
    vae_size = VariationalAutoEncoder.approx_trainable_parameters(vae_sizes)
    approx_size = conv2d_size + vae_size
    print(f"Conv2D={conv2d_size:,}, VAE={vae_size:,}, approx total={approx_size:,}")

    if is_too_large(approx_size, max_params):
        return None, model_text, approx_size, vae_size


    model = CombinedVAE(conv2d, vae_sizes)

    return model, model_text, approx_size, vae_size



def is_incremental(model_name):
    return "Incremental" in model_name

def is_audio(model_name):
    return "audio" in model_name.lower()

def model_uses_STFTs(model_name):
    return not is_audio(model_name)

##########################################################################################
# Top-Level to create models and read hyper-parameters
#

def is_too_large(approx_size, max_params):
    if approx_size > max_params:
            print(f"Model is too large: approx {approx_size:,} parameters vs max={max_params:,}")
            return True
    else:
        return False


def invalid_model(size):
    return None, None, size
    
def make_model(model_type, model_params, max_params, verbose):

    # TODO: Move this code into the individual models!

    match model_type:
        case "STFT_VAE":
            latent_size, depth, ratio = model_params
            model_text = f"{model_type} latent={latent_size}, layers={depth}, ratio={ratio:.2f}"
            print(model_text)
            approx_size = STFTVariationalAutoEncoder.approx_trainable_parameters(freq_buckets, sequence_length, latent_size, depth, ratio)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
                
            model = STFTVariationalAutoEncoder(freq_buckets, sequence_length, latent_size, depth, ratio)
        
        
        case "StepWiseMLP":
            hidden_size, depth, ratio = model_params
            model_text = f"{model_type} control={hidden_size}, depth={depth}, ratio={ratio:.2f}"
            print(model_text)
            approx_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(freq_buckets, hidden_size, depth, ratio)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
                
            model = StepWiseMLPAutoEncoder(freq_buckets, sequence_length, hidden_size, depth, ratio)
           
           
        case "MLP_VAE":
            hidden_size, mlp_depth, mlp_ratio, latent_size, vae_depth, vae_ratio = model_params
            model_text = f"{model_type} hidden={hidden_size}, depth={mlp_depth}, ratio={mlp_ratio:.1f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
            print(model_text)

            vae_sizes = interpolate_layer_sizes(hidden_size * sequence_length, latent_size, vae_depth, vae_ratio)
            print(f"VAE layers={vae_sizes}")
            
            mlp_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(freq_buckets, hidden_size, mlp_depth, mlp_ratio)
            vae_size = VariationalAutoEncoder.approx_trainable_parameters(vae_sizes)
            approx_size = mlp_size + vae_size
            print(f"MLP={mlp_size:,}, VAE={vae_size:,}, approx total={approx_size:,}")

            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
            
            mlp = StepWiseMLPAutoEncoder(freq_buckets, sequence_length, hidden_size, mlp_depth, mlp_ratio)
            
            model = CombinedVAE(mlp, vae_sizes)


        case "MLPVAE_Incremental":
            mlp_name, mlp_params, file_name = get_best_configuration_for_model("StepWiseMLP")
            mlp_params = mlp_params[2:] # remove the optimiser params
            combined_params = mlp_params + model_params # add the VAE params.

            hidden_size, mlp_depth, mlp_ratio, latent_size, vae_depth, vae_ratio = combined_params
            model_text = f"{model_type} hidden={hidden_size}, depth={mlp_depth}, ratio={mlp_ratio:.1f}, latent={latent_size}, VAE depth={vae_depth}, VAE ratio={vae_ratio:.2f}"
            print(model_text)

            vae_sizes = interpolate_layer_sizes(hidden_size * sequence_length, latent_size, vae_depth, vae_ratio)
            print(f"VAE layers={vae_sizes}")
            
            mlp_size = StepWiseMLPAutoEncoder.approx_trainable_parameters(freq_buckets, hidden_size, mlp_depth, mlp_ratio)
            vae_size = VariationalAutoEncoder.approx_trainable_parameters(vae_sizes)
            approx_size = mlp_size + vae_size
            print(f"MLP={mlp_size:,}, VAE={vae_size:,}, approx total={approx_size:,}")

            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
            
            mlp = StepWiseMLPAutoEncoder(freq_buckets, sequence_length, hidden_size, mlp_depth, mlp_ratio)
            
            model = CombinedVAE(mlp, vae_sizes)
            
            # Incremental training: load the previous saved state, and freeze the layers we won't re-train
            load_weights_and_biases(mlp, file_name)
            freeze_model(mlp)
            approx_size = vae_size # we're not re-training the RNN parameters

    
        case "RNNAutoEncoder":
            hidden_size, encode_depth, decode_depth = model_params
            model_text = f"{model_type} hidden={hidden_size}, encode_depth={encode_depth}, decode_depth={decode_depth}"
            print(model_text)
            dropout = 0 # will explore this later.
            approx_size = RNNAutoEncoder.approx_trainable_parameters(freq_buckets, hidden_size, encode_depth, decode_depth)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
                
            model = RNNAutoEncoder(freq_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout)
    
    
        case "RNN_VAE":
            model, model_text, approx_size, vae_size = make_RNN_VAE(model_type, model_params, max_params)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)


        case "RNN_VAE_Incremental": # We load a trained RNN Auto-Encoder, and train a further lever of compression using a VAE.
            rnn_name, rnn_params, file_name = get_best_configuration_for_model("RNNAutoEncoder")
            rnn_params = rnn_params[2:] # remove the optimiser params
            print(f"rnn_params={rnn_params}")
            print(f"model_params={model_params}")
            combined_params = rnn_params + model_params # add the VAE params.
            print(f"combined={combined_params}")
            model, model_text, approx_size, vae_size = make_RNN_VAE(model_type, combined_params, max_params)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
            
            # Incremental training: load the previous saved state, and freeze the layers we won't re-train
            load_weights_and_biases(model.auto_encoder, file_name)
            freeze_model(model.auto_encoder)
            approx_size = vae_size # we're not re-training the RNN parameters


        case "RNN_F&T": # this model refused to train...
            freq_size, freq_depth, time_size, time_depth = [int(x) for x in model_params] # convert int64 to int32
            model_text = f"{model_type} frequency={freq_size} x {freq_depth}, time={time_size} x {time_depth}"
            print(model_text)
            approx_size = RNNFreqAndTime.approx_trainable_parameters(freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)
                
            dropout = 0
            model = RNNFreqAndTime(freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth, dropout)


        case "Conv2D_AE":
            layer_count, kernel_count, kernel_size = [int(x) for x in model_params]  # convert int64 to int32
            model_text = f"{model_type} layer_count={layer_count}, kernel_count={kernel_count}, kernel_size={kernel_size}"
            print(model_text)
            approx_size = Conv2DAutoEncoder.approx_trainable_parameters(layer_count, kernel_count, kernel_size)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)

            model = Conv2DAutoEncoder(freq_buckets, sequence_length, layer_count, kernel_count, kernel_size)
            model.float()
            model.to(device)

            if model.compression < min_compression or model.compression > max_compression:
                print(f"Compression={model.compression:.1f} out of range [{min_compression}, {max_compression}]")
                return invalid_model(approx_size)

        case "Conv2D_VAE_Incremental":
            conv_name, conv_params, file_name = get_best_configuration_for_model("Conv2D_AE")
            conv_params = conv_params[2:]  # remove the optimiser params
            print(f"conv_params={conv_params}")
            print(f"model_params={model_params}")
            combined_params = conv_params + model_params  # add the VAE params.
            print(f"combined={combined_params}")
            model, model_text, approx_size, vae_size = make_Conv2D_VAE(model_type, combined_params, max_params)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)

            # Incremental training: load the previous saved state, and freeze the layers we won't re-train
            load_weights_and_biases(model.auto_encoder, file_name)
            freeze_model(model.auto_encoder)
            approx_size = vae_size  # we're not re-training the Conv2D parameters

        case "AudioConv_AE":
            depth, kernel_count, outer_kernel_size, inner_kernel_size = model_params
            model_text = f"{model_type} layers={depth}, kernel_count={kernel_count}, outer_kernel={outer_kernel_size}, inner_kernel={inner_kernel_size}"
            print(model_text)
            approx_size = AudioConv_AE.approx_trainable_parameters(depth, kernel_count, outer_kernel_size, inner_kernel_size)
            if is_too_large(approx_size, max_params):
                return invalid_model(approx_size)

            model = AudioConv_AE(audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size)

            if model.compression < min_compression or model.compression > max_compression:
                print(f"Compression={model.compression:.1f} out of range [{min_compression}, {max_compression}]")
                return invalid_model(approx_size)

        case _:
            raise Exception(f"Unknown model: {model_type}")


    # Check the real size:
    size = count_trainable_parameters(model)
#    print(f"{model_type} {size:,} parameters")
#    model_text += f" ({size:,} parameters)"
    
    # Warn if the approximation was off:
    size_error = approx_size / size  - 1
    if np.abs(size_error) > 0.01:
        print(f"*** Inaccurate approximate size={approx_size:,} vs actual size={size:,}, error={100*size_error:.2f}%")
    
    # Too big?
    if size > max_params:
        print(f"Model is too large: {size:,} parameters vs max={max_params:,}")
        return invalid_model(approx_size)

    # Get ready!
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    if verbose:
        print("model={}".format(model))
    
    return model, model_text, size


def get_best_configuration_for_model(model_name):
    file_name = "Models/" + model_name
    with open(file_name + ".txt", 'r') as file:
        first_line = file.readline().strip()
        params = ast.literal_eval(first_line)
        print(f"{model_name}: stored params={params}")
        
    return model_name, params, file_name + ".wab"



def load_saved_model(model_name):
    model_type, params, file_name = get_best_configuration_for_model(model_name)
    model_params = params[2:] # remove the optimiser configuration
    max_params = +1e99 # ignore
    verbose = True
    model, model_text, model_size = make_model(model_type, model_params, max_params, verbose)
    
    print(f"Loading weights & biases from file '{file_name}'")
    model.load_state_dict(torch.load(file_name))
    model.eval() # Ensure the model is in evaluation mode
    model.to(device)
    print(f"{model_type} has {count_trainable_parameters(model):,} weights & biases")
    
    return model, model_text, params, model_size

