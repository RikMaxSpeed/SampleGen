# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from Train import *
from AutoEncoderModels import *


# skopt creates some unhelpful warnings...
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

# The huge MLP_VAE needs order of 120M parameters.
#max_params = 1000 * stft_buckets * sequence_length # Approx size of 1000 STFTs in the training set
# But the StepWiseMLP and RNN are much more memory efficient
max_params = stft_buckets * sequence_length

count = 0
break_on_exceptions = True # Set this to False to allow the process to continue even if the model blows up (useful for long tuning runs!)


def evaluate_model(params):
    global count
    count += 1
    print(f"\n\n\nHyper-Parameter tuning#{count}\n")
    
    max_time = 300
    max_overfit = 1.1
    max_epochs = 100 # This is sufficient to figure out which model will converge best if we let it run for longer.
    verbose = False
    
    if break_on_exceptions: # this is easier when debugging
        return train_model(params, max_epochs, max_time, max_params, max_overfit, verbose)
    else: # we need this for long overnight runs in case something weird happens
        try:
            return train_model(params, max_epochs, max_time, max_params, max_overfit, verbose)
        except Exception as e:
            print(f"*** Exception: {e}")
        except:
            print(f"*** Something broke :(")
        
        return 100000 # return a high loss...


def optimise_hyper_parameters():
    count = 200  # use a smaller data-set here to speed things up? Not a good idea as the model may be too limited in size
    count = 1100
    generate_training_stfts(count)
    
    global max_params
    max_params = (stft_buckets * sequence_length * count) // 20
    
    # Optimiser:
    search_space = list()
    search_space.append(Integer(  16,    256,    'log-uniform',  name='batch_size'))
    search_space.append(Real   (1e-6,   1e-2,   'log-uniform',  name='learning_rate'))
    search_space.append(Real   (1e-8,   1e-2,   'log-uniform',  name='weight_decay'))

    # Model:
    if False:
        # Train the STFTVariationalAutoEncoder
        set_model_type("VAE_MLP")
        search_space.append(Integer(4,      8,      'uniform',      name='latent_size'))
        search_space.append(Real   (1.0,    10.0,   'uniform',      name='layer3_ratio'))
        search_space.append(Real   (1.0,    10.0,   'uniform',      name='layer2_ratio'))
        search_space.append(Real   (1.0,    10.0,   'uniform',      name='layer1_ratio'))
        
    if False:
        # Train just the StepWiseMLPAutoEncode (with no VAE)
        set_model_type("StepWiseMLP")
        search_space.append(Integer(10,       50,   'uniform',      name='control_size'))
        search_space.append(Integer(2,         4,   'uniform',      name='depth'))
        search_space.append(Real   (0.1,     2.0,   'log-uniform',  name='ratio'))
        
    if False:
        # Train the StepWiseVAEMLPAutoEncoder
        set_model_type("StepWiseVAEMLP")
        
        # StepWiseMLP parameters
        search_space.append(Integer(40,       50,   'uniform',      name='control_size'))
        search_space.append(Integer(2,         4,   'uniform',      name='depth'))
        search_space.append(Real   (0.1,       4,   'log-uniform',  name='ratio'))
        
        # VAE parameters:
        search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
        search_space.append(Integer(1,         5,   'uniform',      name='vae_depth'))
        search_space.append(Real   (0.1,       4,   'log-uniform',  name='vae_ratio'))
    
    if False:
        # Train the RNNAutoEncoder
        set_model_type("RNNAutoEncoder")
        search_space.append(Integer(10,       60,   'uniform',      name='hidden_size'))
        search_space.append(Integer(1,         4,   'uniform',      name='encode_depth'))
        search_space.append(Integer(1,         4,   'uniform',      name='decode_depth'))

    if True:
        # Train the RNN_VAE
        set_model_type("RNN_VAE")
        # RNN parameters
        search_space.append(Integer(10,       60,   'uniform',      name='hidden_size'))
        search_space.append(Integer(1,         2,   'uniform',      name='encode_depth'))
        search_space.append(Integer(1,         2,   'uniform',      name='decode_depth'))
        
        # VAE parameters
        search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
        search_space.append(Integer(1,         5,   'uniform',      name='vae_depth'))
        search_space.append(Real   (0.1,       4,   'log-uniform',  name='vae_ratio'))



    print("Optimising hyper-parameters:")
    display(search_space)

    result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=16, initial_point_generator='sobol', noise='gaussian', verbose=True)

    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))


best_models = {

    "RNN_VAE": ([18, 0.00016409427656154815, 1.9378132418753713e-05, 24, 1, 1, 6, 2, 3.359267821929004],
    "RNN_VAE hidden=24, encode_depth=1, decode_depth=1, latent=6, VAE depth=2, VAE ratio=3.36.wab"),
    
    "StepWiseVAEMLP": ([28, 0.000257541325218376, 2.002069396636439e-08, 48, 2, 0.5002532704787946, 6, 4, 1.4277438221372638],
    "StepWiseVAEMLP control=48, depth=2, ratio=0.50, latent=6, VAE depth=4, VAE ratio=1.43.wab")
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
    max_overfit = 2.0 # we're aiming for high precision on the training set
    max_params = 1000000000
    max_epochs = 2000 # we don't hit this in practice.
    
    verbose = True
    train_model(params, max_epochs, max_time, max_params, max_overfit, verbose)
    #torch.save(model.state_dict(), model_file) # train_model does this.
