# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

from Train import *
#from MakeSTFTs import *


one_sample = stft_buckets * sequence_length
max_params = None
tuning_count = 0
break_on_exceptions = True # True=Debugging, False allows the GPR to continue even if the model blows up (useful for long tuning runs!)
max_loss = 4000 # default

hyper_losses = []
hyper_names = []
hyper_model = None

    
def evaluate_model(params):
    global hyper_model, hyper_losses, hyper_names, tuning_count
    
    tuning_count += 1
    print(f"Hyper-Parameter tuning#{tuning_count}: {hyper_model} {params}\n")
    
    #max_time = 5 * 60 # seconds
    max_overfit = 1.2 # Ensure we retain the models that generalise reasonably well.
    
    if is_incremental(hyper_model):
        max_overfit = 3.0 # the internal layer may have been deliberately over-fit.
        print(f"Overriding max_overfit={max_overfit:.1f}")
        
    max_epochs = 100 # This is sufficient to figure out which model will converge best if we let it run for longer.
    max_time = 300 # we don't like slow models...
    verbose = False # avoid printing lots of detail for each run
    
    if break_on_exceptions: # this is easier when debugging
        loss, model_text = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
        
    else: # we need this for long overnight runs in case something weird happens
        try:
            loss, model_text = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
            
        except Exception as e:
            print(f"*** Exception: {e}")
            return max_loss
        except:
            print(f"*** Breaking Bad :(")
            return max_loss
        
    if loss < max_loss:
        hyper_losses.append(loss)
        hyper_names.append(model_text)
        plot_hypertrain_loss(hyper_losses, hyper_names, hyper_model)
    
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



def optimise_hyper_parameters(model_name):
    # use a smaller data-set here to speed things up? Not a good idea as the model may be too limited in size
    samples, _ = generate_training_stfts(100)
    print(f"Training data set has {samples} samples.")
    
    global max_params, max_loss
    train_data_size = samples * one_sample
    max_params = 200 * one_sample # encoder & decoder will be approx half that size
    print(f"{stft_buckets} frequencies, {sequence_length} time-steps, maximum model size is {max_params:,} parameters.")
    
    # Optimiser:
    batch = 4 # actual batch_size = 2**batch
    lr = 1e-6
    
    search_space = list()
    search_space.append(Integer(batch,  batch+3,    'uniform',      name='batch')) # batch_size = 2^batch
    search_space.append(Real   (lr,    lr * 100,    'log-uniform',  name='learning_rate')) # scaled by the batch_size

    # Model:
    global hyper_model
    hyper_model = model_name
    
    match model_name:
        
        case "STFT_VAE":
            # Train the naive STFTVariationalAutoEncoder
            max_params = 200_000_000 # this model needs a huge number of parameters
            max_loss = 100_000 # and the loss starts off extremely high
            search_space.append(Integer(4,        10,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         3,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='vae_ratio'))
            
        case "StepWiseMLP":
            # Train just the StepWiseMLPAutoEncode (with no VAE)
            search_space.append(Integer(8,        12,   'uniform',      name='hidden_size'))
            search_space.append(Integer(3,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
        case "StepWiseVAEMLP" | "MLP_VAE":
            # StepWiseMLP parameters
            search_space.append(Integer(8,        12,   'uniform',      name='hidden_size'))
            search_space.append(Integer(3,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
            # VAE parameters:
            search_space.append(Integer(5,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))
        
        case "MLPVAE_Incremental":
            # We only need the VAE parameters, as the StepWiseMLP has already been trained.
            search_space.append(Integer(8,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         7,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,    10.0, 'uniform',        name='vae_ratio'))
            
        case "RNNAutoEncoder": # Train the RNNAutoEncoder (no VAE)
            max_params = 20_000_000
            search_space.append(Integer(50,      80,   'log-uniform',   name='hidden_size'))
            search_space.append(Integer(2,        6,   'uniform',       name='encode_depth'))
            search_space.append(Integer(2,        6,   'uniform',       name='decode_depth'))

        case "RNN_VAE": # Train the full RNN_VAE

            # RNN parameters
            search_space.append(Integer(60,      100,   'log-uniform',      name='hidden_size'))
            search_space.append(Integer(1,         5,   'uniform',      name='encode_depth'))
            search_space.append(Integer(1,         5,   'uniform',      name='decode_depth'))
            
            # VAE parameters
            search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(3,        10,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case "RNN_VAE_Incremental": # Train the VAE only and load a pre-trained RNNAutoEncoder
            # VAE parameters
            search_space.append(Integer(5,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         4,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.2,       5,   'uniform',      name='vae_ratio'))
        
        case _:
            raise Exception(f"Invalid model type = {model_name}")

    print("Optimising hyper-parameters:")
    display(search_space)

    # Generate starting parameters, around the minimum sizes which tend to generate smaller networks
    if False:
        generate = 3 * len(search_space)
        amount = 0.7
        initial_params = [generate_parameters(search_space, amount) for amount in np.arange(0.0, amount, amount/generate)]
        result = gp_minimize(evaluate_model, search_space, n_calls=1000, x0 = initial_params, noise='gaussian', verbose=False, acq_func='LCB')
    else:
        result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=8, initial_point_generator='sobol', noise='gaussian', verbose=False, acq_func='LCB')
        
    # I've never reached this point! :)
    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))





def train_best_params(model_name):
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(None) # No augmentation
    
    model_name, params, _ = get_best_configuration_for_model(model_name)

    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9 # ignore - we want high fidelity on the training set, though this could result in less diversity on the generated audio... Mmm...
    max_epochs = 2000 # we don't hit this in practice.
    max_loss = 1e9
    
    params[0] = 4 # override the batch-size
    
    verbose = True
    train_model(model_name, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
