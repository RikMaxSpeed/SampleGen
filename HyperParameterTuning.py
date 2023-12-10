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
    print(f"Hyper-Parameter tuning#{tuning_count}: [params]\n")
    
    #max_time = 5 * 60 # seconds
    max_overfit = 1.1 # Ensure we retain the models that generalise best.
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
    print(f"{tuning_count} training samples, {stft_buckets} frequencies, {sequence_length} time-steps, maximum model size is {max_params:,} parameters.")
    
    # Optimiser:
    search_space = list()
    search_space.append(Integer(  4,    8,       'uniform',      name='batch')) # batch_size = 2^batch
    search_space.append(Real   (1e-7,   1e-2,    'log-uniform',  name='learning_rate')) # scaled by the batch_size
    search_space.append(Real   (1e-9,   1e-2,    'log-uniform',  name='weight_decay'))

    model_name = 'StepWiseMLP' #"StepWiseVAEMLP"
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
            search_space.append(Integer(10,      200,   'uniform',      name='control_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))
            
        case "StepWiseVAEMLP":
            # Train the StepWiseMLP_VAE
            
            # StepWiseMLP parameters
            search_space.append(Integer(40,      100,   'uniform',      name='control_size'))
            search_space.append(Integer(2,         6,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,       5,   'log-uniform',  name='ratio'))
            
            # VAE parameters:
            search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(1,         7,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='vae_ratio'))
        
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
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='vae_ratio'))

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


best_models = {
#*** Best! loss=791.04
#"StepWiseVAEMLP": ([64, 1e-5, 0.001, 43, 5, 0.13753954871555363, 8, 2, 0.18245971697542837], "No file"),
"StepWiseVAEMLP": ([16, 1.4815996677501001e-05, 0.0016767313292796594, 43, 5, 0.13753954871555363, 8, 2, 0.18245971697542837], "No file"),

    "RNN_VAE": ([18, 0.0005575544181212729, 5.294016993959888e-06, 29, 1, 2, 4, 4, 0.20679719844604053],
                "StepWiseVAEMLP control=48, depth=2, ratio=0.50, latent=6, VAE depth=4, VAE ratio=1.43.wab"), # train loss=0.01244, test  loss=0.01417

#    "RNN_VAE": ([18, 0.00016409427656154815, 1.9378132418753713e-05, 24, 1, 1, 6, 2, 3.359267821929004],
#    "RNN_VAE hidden=24, encode_depth=1, decode_depth=1, latent=6, VAE depth=2, VAE ratio=3.36.wab"),
    
    # This model achieves good losses, however it uses 8 variables in the latent parameter, and at least 3 of them appear highly correlated.
#    "StepWiseVAEMLP": ([16, 0.0008253527686277826, 2.8929226732001846e-06, 45, 4, 2.426466845325152, 8, 1, 0.7256301852268706],
#    "StepWiseVAEMLP control=45, depth=4, ratio=2.43, latent=8, VAE depth=1, VAE ratio=0.73.wab") # train loss=0.00768, test  loss=0.00862
}


def get_best_model_configuration():
    #best = "RNN_VAE"
    best = "StepWiseVAEMLP"
    set_model_type(best)
    return best_models[best]


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


def train_best_params():
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(None) # No augmentation
    
    params, _ = get_best_model_configuration()

    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9 # ignore
    max_epochs = 2000 # we don't hit this in practice.
    max_loss = 1e9
    
    verbose = True
    train_model(params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose)
