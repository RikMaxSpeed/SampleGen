# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from Train import *

# skopt creates some unhelpful warnings...
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

max_params = 120000000 # Approx size of 1000 STFTs in the trainin set

count = 0
break_on_exceptions = True

def evaluate_model(params):
    global count
    count += 1
    print(f"\n\n\nHyper-Parameter tuning#{count}\n")
    
    max_time = 300
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


def optimise_hyper_parameters():
    generate_training_stfts(100) # we could use a smaller data-set here to speed things up?

    # Optimiser:
    search_space = list()
    search_space.append(Integer(16,     512,    'log-uniform',  name='batch_size'))
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
        
    if True:
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
    

#    if False: # didn't work well
#        set_model_type("Hybrid_CNN")
#        search_space.append(Integer(3,      4,      'uniform',      name='kernel_count'))
#        search_space.append(Integer(5,      6,      'uniform',      name='kernel_size'))
#        search_space.append(Integer(30,    31,      'uniform',      name='rnn_hidden_size'))

        
    print("Optimising hyper-parameters:")
    display(search_space)

    result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=16, initial_point_generator='sobol', noise='gaussian')

    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))



def get_best_hyper_params():
    set_model_type("VAE_MLP")
    return [16, 0.0001, 5.151727534054279e-05, 6, 7.753192086063947, 7.411389428825689, 5.301401097330652]
    #return [9, 0.0009685451968313163, 5.151727534054279e-05, 6, 7.753192086063947, 7.411389428825689, 5.301401097330652, ] # Converges too quickly
    #return [9, 4.5017602108986394e-05, 1.767746772905285e-07, 8, 3.7266841851402606, 3.787922442988532, 8.189781767196333, ]
    
    
def get_best_filename():
    return "Model latent=6, layer3=46, layer2=340, layer1=1802, loss=0.0063.wab"
    #return "Model latent=8, layer3=29, layer2=109, layer1=892, loss=0.0026.wab" # Mu=1 (linear, no transform), small latent size


def load_best_model():
    set_model_type("VAE_MLP")
    model = make_model(get_best_hyper_params(), None, True)
    model.load_state_dict(torch.load(get_best_filename()))
    model.eval() # Ensure the model is in evaluation mode
    model.to(device)
    
    return model


def train_best_params():
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(930) # No augmentation
    
    params = get_best_hyper_params()
    
    max_time = 12 * 3600 # we should converge way before this!
    max_overfit = 2.0 # we're aiming for max precision on the training set
    max_params = 1000000000
    
    verbose = True
    train_model(params, max_time, max_params, max_overfit, verbose)
    #torch.save(model.state_dict(), model_file) # train_model does this.
