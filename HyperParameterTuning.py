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
max_params = 10 * stft_buckets * sequence_length

count = 0
break_on_exceptions = False # Set this to False to allow the process to continue even if the model blows up (useful for long tuning runs!)

def evaluate_model(params):
    global count
    count += 1
    print(f"\n\n\nHyper-Parameter tuning#{count}\n")
    
    max_time = 500
    max_ratio = 4.0
    max_overfit = 1.1
    verbose = False
    
    if break_on_exceptions: # this is easier when debugging
        return train_model(params, max_time, max_params, max_overfit, verbose)
    else: # we need this for long overnight runs in case something weird happens
        try:
            return train_model(params, max_time, max_params, max_overfit, verbose)
        except Exception as e:
            print(f"*** Exception: {e}")
        except:
            print(f"*** Something broke :(")
        
        return 100000 # return a high loss...


def optimise_hyper_parameters():
    generate_training_stfts(200) # use a smaller data-set here to speed things up?

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

    result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=16, initial_point_generator='sobol', noise='gaussian')

    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))



def get_best_hyper_params():
    set_model_type("StepWiseVAEMLP")
    return [28, 0.000257541325218376, 2.002069396636439e-08, 48, 2, 0.5002532704787946, 6, 4, 1.4277438221372638]
    #return [32, 0.0003778450201583826, 3.4703673500392346e-08, 42, 3, 0.5642073310609655, 6, 4, 0.5032937921364286]
    
#    set_model_type("VAE_MLP")
#    return [16, 0.0001, 5.151727534054279e-05, 6, 7.753192086063947, 7.411389428825689, 5.301401097330652]
    
    
def get_best_file_name():
    return "StepWiseVAEMLP control=48, depth=2, ratio=0.50, latent=6, VAE depth=4, VAE ratio=1.43.wab"
    #return "StepWiseVAEMLP control=42, depth=3, ratio=0.56, latent=6, VAE depth=4, VAE ratio=0.50.wab"
    #return "Model latent=6, layer3=46, layer2=340, layer1=1802, loss=0.0063.wab"
    #return "Model latent=8, layer3=29, layer2=109, layer1=892, loss=0.0026.wab" # Mu=1 (linear, no transform), small latent size


def load_best_model():
    model_params = get_best_hyper_params()[3:]
    max_params = 1000000000
    verbose = True
    model, model_text = make_model(model_params, max_params, verbose)
    file_name = get_best_file_name()
    
    print(f"Loading weights & biases from file '{file_name}'")
    model.load_state_dict(torch.load(file_name))
    model.eval() # Ensure the model is in evaluation mode
    model.to(device)
    print(f"{model.__class__.__name__} has {count_trainable_parameters(model):,} weights & biases")
    
    return model


def train_best_params():
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(None) # No augmentation
    
    params = get_best_hyper_params()

    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 2.0 # we're aiming for high precision on the training set
    max_params = 1000000000
    
    verbose = True
    train_model(params, max_time, max_params, max_overfit, verbose)
    #torch.save(model.state_dict(), model_file) # train_model does this.
