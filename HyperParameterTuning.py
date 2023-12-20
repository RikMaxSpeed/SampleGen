# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.
from appscript.terminology import params
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

max_params = 0
tuning_count = 0
break_on_exceptions = True # True=Debugging, False allows the GPR to continue even if the model blows up (useful for long tuning runs!)
max_loss = 10_000 # default

hyper_model = "None"
hyper_losses = []
hyper_names  = []
hyper_params = []

def reset_hyper_training():
    global hyper_model, hyper_losses, hyper_names, hyper_params
    hyper_model = "None"
    hyper_losses = []
    hyper_names = []
    hyper_params = []
    reset_train_losses()


from Train import *

one_sample = freq_buckets * sequence_length
print(f"1 sample = {freq_buckets:,} x {sequence_length:,} = {one_sample:,}")


def evaluate_model(params):
    global hyper_model, hyper_losses, hyper_names, hyper_epochs, tuning_count
    
    tuning_count += 1
    print(f"Hyper-Parameter tuning#{tuning_count}: {hyper_model} {params}\n")
    
    #max_time = 5 * 60 # seconds
    max_overfit = 1.5 # Ensure we retain the models that generalise reasonably well.
    
    if is_incremental(hyper_model):
        max_overfit = 3.0 # the internal layer may have been deliberately over-fit.
        print(f"Overriding max_overfit={max_overfit:.1f}")
        
    max_epochs = 80 # This is sufficient to figure out which model will converge best if we let it run for longer.
    if is_incremental_vae(hyper_model):
        max_epochs = 500

    max_time = int(hour/3)  # we don't like slow models...
    verbose = False # avoid printing lots of detail for each run
    preload = False

    if break_on_exceptions: # this is easier when debugging
        model_text, model_size, loss, rate = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, preload)

    else: # we need this for long overnight runs in case something weird happens
        try:
            model_text, model_size, loss, rate = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, preload)

        except Exception as e:
            print(f"*** Exception: {e}")
            return max_loss
        except:
            print(f"*** Breaking Bad :(")
            return max_loss

    if loss < max_loss: # skip the models that failed in some way.
        # Bake the learning_rate into the loss:
        final_loss = loss
        if rate < 0:
            final_loss *= (1 + rate) ** 20 # favourise models that have more scope to improve
        print(f"adjusted final loss={final_loss:.2f}, from loss={loss:.2f} and rate={rate*100:.3f}%")

        hyper_losses.append(final_loss)
        hyper_names.append(model_text + f" | real loss={loss:.2f}, rate={rate * 100:.3f}%") # Record the final learning rate
        hyper_params.append(params)

        order = np.argsort(hyper_losses)
        topN = min(20, len(hyper_losses))

        print("Best hyper parameters:")
        for i in range(topN):
            o = order[i]
            print(f"#{i+1} {hyper_losses[o]:.2f}, {hyper_names[o]}")
        print("\n")

        file_name = hyper_model + " hyper parameters.txt"
        with open(file_name, 'w') as file:
            for i in range(topN):
                o = order[i]
                file.write(str(hyper_params[o]) + "\n")

            file.write("\n\n")
            for i in range(topN):
                o = order[i]
                file.write(f"#{i+1} {hyper_losses[o]:.2f}, {hyper_names[o]}\n")

        if is_interactive:
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
    print(f"\n\n\nOptimising Hyper Parameters for {model_name}\n")
    reset_hyper_training()

    # Use a smaller data-set here to speed things up? Could favour small models that can't handle the entire data-set.
    samples, _ = generate_training_stfts(None) # full data-set, this may be more representative
    #samples, _ = generate_training_stfts(200) # 80% = 10 x batch=16
    print(f"Training data set has {samples} samples.")

    global max_params, max_loss
    train_data_size = samples * one_sample
    max_params = 200 * one_sample # encoder & decoder will be approx half that size
    print(f"{freq_buckets} frequencies, {sequence_length} time-steps, sample={one_sample:,}, maximum model size is {max_params:,} parameters.")

    # Optimiser:
    search_space = list()
    search_space.append(Integer(4,      6,    'log-uniform',  name='batch'))         # batch_size = 2^batch
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
            search_space.append(Integer(10,       40,   'uniform',      name='hidden_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))

        case "MLP_VAE":
            max_loss = 50_000

            # StepWiseMLP parameters
            search_space.append(Integer(10,       40,   'uniform',      name='hidden_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='depth'))
            search_space.append(Real   (0.1,      10,   'log-uniform',  name='ratio'))

            # VAE parameters:
            search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case "MLPVAE_Incremental":
            # We only need the VAE parameters, as the StepWiseMLP has already been trained.
            search_space.append(Integer(4,         8,   'uniform',      name='latent_size'))
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
    print(search_space)
    start_new_stft_video(f"STFT - hyper-train {model_name}", True)

    # Generate starting parameters, around the minimum sizes which tend to generate smaller networks
    max_loops = 60 # it frequently gets stuck at some local minimum before this.
    result = gp_minimize(evaluate_model, search_space, n_calls=max_loops, noise='gaussian', verbose=False, acq_func='LCB')
    #n_initial_points=8, initial_point_generator='sobol',

    # I've never reached this point! :)
    print("\n\nHyper Parameter Optimisation Done!!")
    print(f"Best result={result.fun:.2f}")
    print(f"Best parameters={result.x}")


# Load an existing model and see whether we can further train
def fine_tune(model_name):
    model_type, params, file_name = get_best_configuration_for_model(model_name)
    train_best_params(model_name, params, finest=True)

def train_best_params(model_name, params = None, finest = False):
    if finest:
        print(f"\n\n\nFine-Tuning {model_name}\n")
    else:
        print(f"\n\n\nTraining model {model_name}\n")

    reset_train_losses()

    #generate_training_stfts(200) # Small dataset of the most diverse samples
    generate_training_stfts(None) # Full dataset with no augmentation
    #generate_training_stfts(3000) # use a large number of samples with augmentation

    if params is None:
        model_name, params, _ = get_best_configuration_for_model(model_name)

    print(f"train_best_params: {model_name}: {params}")
    start_new_stft_video(f"STFT - train {model_name}", True)

    max_time = 12 * hour # hopefully the model converges way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9  # not relevant, we have a valid model
    max_epochs = 9999 # ignore
    max_loss = 1e9

     # This does improve the final accuracy, but it's very slow.
    if finest:
        params[0] =  0 # override the batch-size
        params[1] = -6 # override the learning rate
        #max_time = hour

    #set_display_hiddens(True) # Displays the internal auto-encoder output

    verbose = True
    train_model(model_name, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, finest)


def full_hypertrain(model_name):
    optimise_hyper_parameters(model_name)
    train_best_params(model_name)
    fine_tune(model_name)

def grid_search():
    global hyper_model, max_params
    hyper_model = "MLPVAE_Incremental"
    max_params = 20_000_000
    samples, _ = generate_training_stfts(None)  # full data-set, this may be more representative
    params = [4, -5, 4, 2, 0.1] # this is where gp_minimize got stuck...
    for vae_depth in range(2, 5):
        for latent in range(4, 8):
            params[2] = latent
            params[3] = vae_depth
            print(f"\n\n\nGrid Hyper-train: latent={latent}, vae_depth={vae_depth}")
            evaluate_model(params)

if __name__ == '__main__':
    # Edit this to perform whatever operation is required.
    
    # MLP VAE model
    #full_hypertrain("StepWiseMLP")
    #full_hypertrain("MLPVAE_Incremental") # Gets stuck in at a local minimum...

    train_best_params("StepWiseMLP", [3, -5, 35, 3, 1.0]) # small batches converge faster!!
    #fine_tune("StepWiseMLP")
    train_best_params("MLPVAE_Incremental", [3, -5, 6, 4, 1])
    fine_tune("MLPVAE_Incremental")

    #optimise_hyper_parameters("MLPVAE_Incremental") # finds the optimal config and keeps looping over that...

    #full_hypertrain("MLP_VAE") # Just to see whether this stands a chance of working
    #train_best_params("MLP_VAE", [4, -5, 32, 2, 0.2, 7, 4, 0.1])

    #grid_search()
    # train_best_params("StepWiseMLP")
    # fine_tune("StepWiseMLP")
    # train_best_params("MLPVAE_Incremental")
    # fine_tune("MLPVAE_Incremental")

    # RNN VAE model
    #full_hypertrain("RNNAutoEncoder")
    #full_hypertrain("RNN_VAE_Incremental")

    
