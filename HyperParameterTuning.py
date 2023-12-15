# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

from Train import *

one_sample = freq_buckets * sequence_length
print(f"1 sample = {freq_buckets:,} x {sequence_length:,} = {one_sample:,}")

max_params = None
tuning_count = 0
break_on_exceptions = True # True=Debugging, False allows the GPR to continue even if the model blows up (useful for long tuning runs!)
max_loss = 10_000 # default

hyper_losses = []
hyper_names = []
hyper_model = None
    
    
def evaluate_model(params):
    global hyper_model, hyper_losses, hyper_names, tuning_count
    
    tuning_count += 1
    print(f"Hyper-Parameter tuning#{tuning_count}: {hyper_model} {params}\n")
    
    #max_time = 5 * 60 # seconds
    max_overfit = 1.5 # Ensure we retain the models that generalise reasonably well.
    
    if is_incremental(hyper_model):
        max_overfit = 3.0 # the internal layer may have been deliberately over-fit.
        print(f"Overriding max_overfit={max_overfit:.1f}")
        
    max_epochs = 100 # This is sufficient to figure out which model will converge best if we let it run for longer.
    max_time = 600 # we don't like slow models...
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
    # use a smaller data-set here to speed things up? Could favour small models that can't handle the entire data-set.
    #samples, _ = generate_training_stfts(None)
    samples, _ = generate_training_stfts(160) # 10 x batch=16
    print(f"Training data set has {samples} samples.")
    
    global max_params, max_loss
    train_data_size = samples * one_sample
    max_params = 200 * one_sample # encoder & decoder will be approx half that size
    print(f"{freq_buckets} frequencies, {sequence_length} time-steps, sample={one_sample:,}, maximum model size is {max_params:,} parameters.")
    
    # Optimiser:
    search_space = list()
    search_space.append(Integer(3,      6,    'log-uniform',  name='batch'))         # batch_size = 2^batch
    search_space.append(Integer(-7,    -3,    'uniform',  name='learning_rate')) # 10^lr * batch_size

    # Model:
    global hyper_model
    hyper_model = model_name
    
    match model_name:
        
        case "STFT_VAE":
            # Train the naive STFTVariationalAutoEncoder
            max_params = 50_000_000 # this model needs a huge number of parameters
            max_loss = 300_000 # and the loss starts off extremely high
            max_overfit = 10
            search_space.append(Integer(4,        7,    'uniform',      name='latent_size'))
            search_space.append(Integer(1,        4,    'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='vae_ratio'))
            
        case "StepWiseMLP":
            # Train just the StepWiseMLPAutoEncode (with no VAE)
            search_space.append(Integer(10,       50,   'uniform',      name='hidden_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
        case "MLP_VAE":
            max_loss = 50_000
            
            # StepWiseMLP parameters
            search_space.append(Integer(20,       50,   'uniform',      name='hidden_size'))
            search_space.append(Integer(3,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
            # VAE parameters:
            search_space.append(Integer(5,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',      name='vae_ratio'))
        
        case "MLPVAE_Incremental":
            # We only need the VAE parameters, as the StepWiseMLP has already been trained.
            search_space.append(Integer(5,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))
            
        case "RNNAutoEncoder": # Train the RNNAutoEncoder (no VAE)
            max_loss = 20_000
            search_space.append(Integer(10,      50,   'log-uniform',   name='hidden_size'))
            search_space.append(Integer(2,        6,   'uniform',       name='encode_depth'))
            search_space.append(Integer(2,        6,   'uniform',       name='decode_depth'))

        case "RNN_VAE": # Train the full RNN_VAE
            # RNN parameters
            search_space.append(Integer(10,      50,   'log-uniform',   name='hidden_size'))
            search_space.append(Integer(2,        6,   'uniform',       name='encode_depth'))
            search_space.append(Integer(2,        6,   'uniform',       name='decode_depth'))
            
            # VAE parameters:
            search_space.append(Integer(5,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case "RNN_VAE_Incremental": # Train the VAE only and load a pre-trained RNNAutoEncoder
            # We only need the VAE parameters, as the StepWiseMLP has already been trained.
            search_space.append(Integer(5,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))
        
        case "RNN_F&T":
            search_space.append(Integer(10,     100,   'log-uniform',       name='freq_size'))
            search_space.append(Integer(1,        3,   'log-uniform',       name='freq_depth'))
            search_space.append(Integer(10,      40,   'log-uniform',       name='time_size'))
            search_space.append(Integer(1,        3,   'log-uniform',       name='time_depth'))

        
        case _:
            raise Exception(f"Invalid model type = {model_name}")

    print("Optimising hyper-parameters:")
    display(search_space)
    start_new_stft_video(f"STFT - hyper-train {model_name}")

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





def train_best_params(model_name, params = None):
    #generate_training_stfts(100) # Small dataset of the most diverse samples
    generate_training_stfts(None) # Full dataset with no augmentation
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    
    if params is None:
        model_name, params, _ = get_best_configuration_for_model(model_name)
        
    print(f"train_best_params: {model_name}: {params}")
    start_new_stft_video(f"STFT - train {model_name}")
    
    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9 # ignore - we want high fidelity on the training set, though this could result in less diversity on the generated audio... Mmm...
    max_epochs = 2000 # we don't hit this in practice.
    max_loss = 1e9
    
#    params[0] =  2 # override the batch-size
#    params[1] = -7 # override the learning rate
    
    #set_display_hiddens(True) # Displays the internal auto-encoder output
    
    verbose = True
    train_model(model_name, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)



if __name__ == '__main__':
    # Edit this to perform whatever operation is required.
    
    #config = "StepWiseMLP", None # hyper-optimised
    #config = "MLP_VAE", None
    #config = "MLPVAE_Incremental", None
    config = "StepWiseMLP", [3, -5, 35, 4, 0.1] # hand-specified
    
    model_name, params = config
    
    #optimise_hyper_parameters(model_name)
    train_best_params(model_name, params)

