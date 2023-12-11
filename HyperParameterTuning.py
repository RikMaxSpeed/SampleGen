# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

from Train import *
#from MakeSTFTs import *


one_sample = stft_buckets * sequence_length
max_params = None
tuning_count = 0
break_on_exceptions = True # Set this to False to allow the process to continue even if the model blows up (useful for long tuning runs!)
max_loss = 30000 # default

hyper_losses = []
hyper_names = []

def evaluate_model(params):
    global tuning_count
    tuning_count += 1
    print(f"Hyper-Parameter tuning#{tuning_count}: {params}\n")
    
    #max_time = 5 * 60 # seconds
    max_overfit = 1.3 # Ensure we retain the models that generalise reasonably well.
    max_epochs = 100 # This is sufficient to figure out which model will converge best if we let it run for longer.
    max_time = max_epochs * 20 # we hopefully won't bump into this.
    verbose = False # avoid printing lots of detail for each run
    
    if break_on_exceptions: # this is easier when debugging
        loss, model_text = train_model(params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
        
    else: # we need this for long overnight runs in case something weird happens
        try:
            loss, model_text = train_model(params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
            
        except Exception as e:
            print(f"*** Exception: {e}")
            return max_loss
        except:
            print(f"*** Breaking Bad :(")
            return max_loss
        
    global hyper_losses
    if loss < max_loss:
        hyper_losses.append(loss)
        hyper_names.append(model_text)
        plot_hypertrain_loss(hyper_losses, hyper_names)
    
    return loss


def generate_parameters(search_space, amount):
    def random_parameter(min_value, max_value, amount):
        t = amount * np.random.uniform()
        return min_value + t * (max_value - min_value)
        
    generated_params = []

    for parameter in search_space:
        min_value, max_value = parameter.bounds
        
        if isinstance(parameter, Real) and parameter.prior == 'log-uniform':
            value = np.exp(random_parameter(np.log(min_value), np.log(max_value), amount))
        else:
            value = random_parameter(min_value, max_value, amount)

        if isinstance(parameter, Integer):
            value = int(round(value))

        value = min(value, max_value)
        value = max(value, min_value)
        
        generated_params.append(value)

    print(f"amount={amount:.2f} -> parameters={generated_params}")
    return generated_params


def optimise_hyper_parameters():
    #tuning_count = 200  # use a smaller data-set here to speed things up? Not a good idea as the model may be too limited in size
    samples = 950
    generate_training_stfts(samples)
    
    global max_params, max_loss
    train_data_size = samples * one_sample
    max_params = train_data_size // 5 # that means both the encode & decoder are approx half that size
    print(f"{stft_buckets} frequencies, {sequence_length} time-steps, maximum model size is {max_params:,} parameters.")
    
    # Optimiser:
    search_space = list()
    search_space.append(Integer(  2,    4,       'uniform',      name='batch')) # batch_size = 2^batch
    search_space.append(Real   (1e-7,   1e-2,    'log-uniform',  name='learning_rate')) # scaled by the batch_size
    search_space.append(Real   (1e-9,   1e-2,    'log-uniform',  name='weight_decay'))

    model_name = "Incremental_StepWiseVAEMLP" #"StepWiseVAEMLP" #"StepWiseMLP" #"StepWiseVAEMLP"
    set_model_type(model_name)

    # Model:
    match model_name:
        
        case "STFT_VAE":
            # Train the naive STFTVariationalAutoEncoder
            max_params = 10*train_data_size # this model needs way more parameters.
            max_loss = 1e7
            search_space.append(Integer(4,       500,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='vae_ratio'))
            
        case "StepWiseMLP":
            # Train just the StepWiseMLPAutoEncode (with no VAE)
            search_space.append(Integer(10,      200,   'uniform',      name='hidden_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
        case "StepWiseVAEMLP":
            # Train the StepWiseMLP_VAE
            max_params = train_data_size
            
            # StepWiseMLP parameters
            search_space.append(Integer(140,      180,   'uniform',     name='hidden_size'))
            search_space.append(Integer(3,         7,   'uniform',      name='depth'))
            search_space.append(Real   (0.2,       5,   'uniform',      name='ratio'))
            
            # VAE parameters:
            search_space.append(Integer(4,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.2,       5,   'uniform',      name='vae_ratio'))
        
        case "Incremental_StepWiseVAEMLP":
            # VAE parameters:
            search_space.append(Integer(4,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.2,       5,   'uniform',      name='vae_ratio'))
                    
        case "RNNAutoEncoder": # Train the RNNAutoEncoder (no VAE)
            search_space.append(Integer(10,     200,   'uniform',      name='hidden_size'))
            search_space.append(Integer(1,        7,   'uniform',       name='encode_depth'))
            search_space.append(Integer(1,        7,   'uniform',       name='decode_depth'))

        case "RNN_VAE": # Train the full RNN_VAE

            # RNN parameters
            search_space.append(Integer(10,      300,   'uniform',      name='hidden_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='encode_depth'))
            search_space.append(Integer(1,         7,   'uniform',      name='decode_depth'))
            
            # VAE parameters
            search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,        10,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case _:
            raise Exception(f"Invalid model type = {model_name}")

    print("Optimising hyper-parameters:")
    display(search_space)

    # Generate starting parameters, around the minimum sizes which tend to generate smaller networks
    if False:
        generate = 3 * len(search_space)
        amount = 0.7
        initial_params = [generate_parameters(search_space, amount) for amount in np.arange(0.0, amount, amount/generate)]
        result = gp_minimize(evaluate_model, search_space, n_calls=1000, x0 = initial_params, noise='gaussian', verbose=False)
    else:
        result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=8, initial_point_generator='sobol', noise='gaussian', verbose=False)

    # I've never reached this point! :)
    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))





def train_best_params():
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(None) # No augmentation
    
    params, _ = get_best_model_configuration()

    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9 # ignore - we want high fidelity on the training set, though this could result in less diversity on the generated audio... Mmm...
    max_epochs = 2000 # we don't hit this in practice.
    max_loss = 1e9
    
    verbose = True
    train_model(params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
