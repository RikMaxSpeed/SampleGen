# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from Train import *

# skopt creates some unhelpful warnings...
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

count = 0

def evaluate_model(params):
    global count
    count += 1
    print(f"\n\n\nHyper-Parameter tuning#{count}\n")
    
    max_time = 300
    max_ratio = 4.0
    max_overfit = 1.1
    verbose = False
    try:
        return train_model(params, max_time, max_ratio, max_overfit, verbose)
    except Exception as e:
        print(f"*** Exception: {e}")
    except:
        print(f"*** Something broke :(")


def optimise_hyper_parameters():
    generate_training_stfts(1000) # we could use a smaller data-set here to speed things up?
    
    search_space = list()
    search_space.append(Integer(4,      8,     'uniform',     name='latent_size'))
    search_space.append(Real   (1.0,    10.0,   'uniform',     name='layer3_ratio'))
    search_space.append(Real   (1.0,    10.0,   'uniform',     name='layer2_ratio'))
    search_space.append(Real   (1.0,    10.0,   'uniform',     name='layer1_ratio'))
    search_space.append(Integer(16,     512,    'log-uniform', name='batch_size'))
    search_space.append(Real   (1e-6,   1e-2,   'log-uniform', name='learning_rate'))
    search_space.append(Real   (1e-8,   1e-2,   'log-uniform', name='weight_decay'))

    print("Optimising hyper-parameters:")
    display(search_space)

    result = gp_minimize(evaluate_model, search_space, n_calls=1000, n_initial_points=32, initial_point_generator='sobol', noise='gaussian')

    print("\n\nHyper Parameter Optimisation Done!!")
    print("Best result={:.2f}".format(result.fun))
    print("Best parameters={}".format(result.x))



#train_model: hyper-parameters=[20, 1.0, 3.995015837017725, 8.892991528946677, 16, 0.0001, 1e-06]
#layers=[116736, 702, 79, 20, 20] -> approx model size=164,013,862 parameters
#hyper-parameters:
#	model: latent=20, layer3=20, layer2=79, layer1=702
#	adam: batch=16, learning_rate=0.0001, weight_decay=1e-06
#STFTVariationalAutoEncoder: sequence_length=114, stft_buckets=1024, sizes=[116736, 702, 79, 20, 20], activation_fn=<class 'torch.nn.modules.activation.ReLU'>
#train=116,736,000 vs approx=164,013,862, ratio=0.71
#164,130,998 trainable parameters vs 116,736,000 inputs, ratio=1.4 to 1
#total=301 sec, epoch=87 (3.5 sec/epoch), train=0.0121 (-1.54%), test=0.0138 (-1.60%), overfit=1.13

# The gp_minimise favours small batch sizes (slow) and learning rates.
# The best accuracy is coming from the smaller models (which presumably converge more easily).

#train_model: hyper-parameters=[24, 2.633160195780127, 2.4705067351395744, 8.822148985996275, 16, 0.00011155383009771261, 2.355823530162682e-06]
#layers=[116736, 1367, 155, 63, 24] -> approx model size=319,605,766 parameters
#hyper-parameters:
#	model: latent=24, layer3=63, layer2=155, layer1=1367
#	adam: batch=16, learning_rate=0.0001, weight_decay=2e-06
#STFTVariationalAutoEncoder: sequence_length=114, stft_buckets=1024, sizes=[116736, 1367, 155, 63, 24], activation_fn=<class 'torch.nn.modules.activation.ReLU'>
#train=116,736,000 vs approx=319,605,766, ratio=0.37
#319,724,014 trainable parameters vs 116,736,000 inputs, ratio=2.7 to 1
#total=306 sec, epoch=51 (6.0 sec/epoch), train=0.0116 (-0.13%), test=0.0127 (-2.83%), overfit=1.09


#train_model: hyper-parameters=[23, 5.435290526226804, 8.44430896429865, 1.4118685699620968, 16, 0.0001, 1e-06]
#layers=[116736, 1489, 1055, 125, 23] -> approx model size=351,056,482 parameters
#hyper-parameters:
#	model: latent=23, layer3=125, layer2=1055, layer1=1489
#	adam: batch=16, learning_rate=0.0001, weight_decay=1e-06
#STFTVariationalAutoEncoder: sequence_length=114, stft_buckets=1024, sizes=[116736, 1489, 1055, 125, 23], activation_fn=<class 'torch.nn.modules.activation.ReLU'>
#train=116,736,000 vs approx=351,056,482, ratio=0.33
#351,176,093 trainable parameters vs 116,736,000 inputs, ratio=3.0 to 1
#total=303 sec, epoch=42 (7.2 sec/epoch), train=0.0114 (2.13%), test=0.0123 (0.87%), overfit=1.09

# Small latent size:
#*** Best! loss=0.0043, model=latent=8, layer3=29, layer2=109, layer1=892, hyper=batch=9, learning_rate=5e-05, weight_decay=2e-07


def train_best_params():
    #generate_training_stfts(3000) # use a large number of samples with augmentation
    generate_training_stfts(930) # No augmentation
    
    params = get_best_hyper_params()
    
    max_time = 12 * 3600 # we should converge way before this!
    max_ratio = 4.0 # not relevant
    max_overfit = 2.0 # we're aiming for max precision on the training set
    
    verbose = True
    train_model(params, max_time, max_ratio, max_overfit, verbose)
    #torch.save(model.state_dict(), model_file) # train_model does this.
