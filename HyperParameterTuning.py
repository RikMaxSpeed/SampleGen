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
max_hyper_runs = 200  # it usually gets stuck at some local minimum well before this.
hyper_stfts = False


def reset_hyper_training(model_name):
    global hyper_model, hyper_stfts, hyper_losses, hyper_names, hyper_params
    hyper_model = model_name
    hyper_stfts = model_uses_STFTs(model_name)
    hyper_losses = []
    hyper_names = []
    hyper_params = []
    reset_train_losses(model_name)

    if is_audio(model_name):
        set_fail_loss(15000)
    else:
        set_fail_loss(1000)


from Train import *

one_sample = freq_buckets * sequence_length
print(f"1 sample = {freq_buckets:,} x {sequence_length:,} = {one_sample:,}")

def display_best_hyper_parameters():

    print("\nBest hyper parameters:")
    order = np.argsort(hyper_losses)
    topN = min(15, len(hyper_losses))
    for i in range(topN):
        o = order[i]
        print(f"#{i + 1} {hyper_losses[o]:.2f}, {hyper_names[o]}")
    print("\n")

    file_name = "Models/" + hyper_model + " hyper parameters.txt"
    with open(file_name, 'w') as file:
        for i in range(topN):
            o = order[i]
            file.write(str(hyper_params[o]) + "\n")

        file.write("\n\n")
        for i in range(topN):
            o = order[i]
            file.write(f"#{i + 1} {hyper_losses[o]:.2f}, {hyper_names[o]}\n")

    if is_interactive:
        plot_hypertrain_loss(hyper_losses, hyper_names, hyper_model)


def evaluate_model(params):
    global hyper_model, hyper_losses, hyper_names, hyper_epochs, tuning_count
    
    tuning_count += 1
    print(f"\n\nHyper-Parameter tuning#{tuning_count}/{max_hyper_runs}: {hyper_model} {params}")
    
    max_overfit = 1.4 # Ensure we retain the models that generalise reasonably well.
    
    max_epochs = 80 # This is sufficient to figure out which model will converge best if we let it run for longer.
    if is_incremental_vae(hyper_model) or is_audio(hyper_model):
        max_epochs = 500 # training the VAE is very fast

    max_time = 5*60 # we don't like slow models...
    verbose = False # avoid printing lots of detail for each run
    preload = False

    if break_on_exceptions: # this is easier when debugging
        model_text, model_size, loss, rate = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, preload)

    else: # we need this for long overnight runs in case something weird happens
        try:
            model_text, model_size, loss, rate = train_model(hyper_model, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, preload)

        except BaseException as e:
            print(f"*** Exception: {e}")
            return max_loss
        except:
            print(f"*** Breaking Bad :(")
            return max_loss

    final_loss = loss
    if loss < get_fail_loss(): # skip the models that failed in some way.
        # Bake the learning_rate into the loss:
        if -0.05 < rate < 0: # only do this if the rate appears plausible
            final_loss *= (1 + rate) ** 30 # favourise models that have more scope to improve
        print(f"adjusted final loss={final_loss:.2f}, from loss={loss:.2f} and rate={rate*100:.3f}%")

        hyper_losses.append(final_loss)
        hyper_names.append(model_text + f" | real loss={loss:.2f}, rate={rate * 100:.3f}%") # Record the final learning rate
        hyper_params.append(params)

        display_best_hyper_parameters()

    return final_loss


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
    reset_hyper_training(model_name)

    # Use a smaller data-set here to speed things up? Could favour small models that can't handle the entire data-set.
    samples, _ = generate_training_data(None, hyper_stfts) # full data-set, this may be more representative
    #samples, _ = generate_training_stfts(200, hyper_stfts) # 80% = 10 x batch=16
    print(f"Training data set has {samples} samples.")

    global max_params, max_loss
    train_data_size = samples * one_sample
    max_params = 200 * one_sample # encoder & decoder will be approx half that size
    print(f"{freq_buckets} frequencies, {sequence_length} time-steps, sample={one_sample:,}, maximum model size is {max_params:,} parameters.")

    # Optimiser:
    search_space = list()
    search_space.append(Integer(2,      6,    'log-uniform',  name='batch'))         # batch_size = 2^batch
    search_space.append(Integer(-7,    -3,    'uniform',  name='learning_rate')) # 10^lr * batch_size

    # Model:
    global hyper_model
    hyper_model = model_name

    # TODO: move this to the individual models.
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

        case "Conv2D_AE":
            max_loss = 30_000
            search_space.append(Integer(3,        6,   'uniform',       name='layer_count'))
            search_space.append(Integer(1,       30,   'uniform',       name='kernel_count'))
            search_space.append(Integer(2,       10,   'uniform',       name='kernel_size'))

        case "Conv2D_VAE_Incremental":
            max_loss = 50_000
            search_space.append(Integer(5,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case "AudioConv_AE":
            # audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size
            max_loss = audio_length
            search_space.append(Integer( 3,     5,     'uniform',  name='layers'))
            search_space.append(Integer(25,    50,     'uniform',  name='kernels'))
            search_space.append(Integer(30,    90, 'log-uniform',  name='outer_kernel_size'))
            search_space.append(Integer( 8,    35, 'log-uniform',  name='inner_kernel_size'))

        case _:
            raise Exception(f"Invalid model type = {model_name}")

    print("Optimising hyper-parameters:")
    print(search_space)
    start_new_stft_video(f"STFT - hyper-train {model_name}", True)

    # Generate starting parameters, around the minimum sizes which tend to generate smaller networks
    result = gp_minimize(evaluate_model, search_space, n_calls=max_hyper_runs, noise='gaussian', verbose=False, acq_func='LCB', kappa=1.0)
    #n_initial_points=8, initial_point_generator='sobol',

    # I've never reached this point! :)
    print("\n\nHyper Parameter Optimisation Done!!")
    print(f"Best result={result.fun:.2f}")
    print(f"Best parameters={result.x}")
    print("\n\n\n")

# Load an existing model and see whether we can further train to the extreme
def fine_tune(model_name):
    model_type, params, file_name = get_best_configuration_for_model(model_name)
    train_best_params(model_name, params, finest=True)

# Load an existing model and train it normally
def resume_train(model_name):
    model_type, params, file_name = get_best_configuration_for_model(model_name)
    params[0] = 5 # reset the batch size as it may be 0 due to the fine tuning
    train_best_params(model_name, params, finest=False)


def train_best_params(model_name, params = None, finest = False):
    if finest:
        print(f"\n\n\nFine-Tuning {model_name}\n")
    else:
        print(f"\n\n\nTraining model {model_name}\n")

    reset_train_losses(model_name)

    #generate_training_stfts(200) # Small dataset of the most diverse samples
    generate_training_data(None, hyper_stfts) # Full dataset with no augmentation
    #generate_training_stfts(3000) # use a large number of samples with augmentation

    if params is None:
        model_name, params, _ = get_best_configuration_for_model(model_name)

    print(f"train_best_params: {model_name}: {params}")
    start_new_stft_video(f"STFT - train {model_name}", True)

    max_time = 12 * hour # hopefully the model converges way before this!
    max_overfit = 100.0 # ignore: we're aiming for the highest precision possible on the training set
    max_params = 1e9  # not relevant, we have a valid model
    max_epochs = 9999 # ignore
    max_loss = 1e6

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


def train_topN_hyper_params(topN = 10):
    # Train the top N but with longer time spans
    order = np.argsort(hyper_losses)
    for i in range(topN):
        n = order[i]
        print(f"\n\n\n\nTraining optimised parameters rank #{i+1}: {hyper_names[n]}")
        params = hyper_params[n]
        params[0] = 2  # batch-size = 2^N
        params[1] = -7 # learning rate = 10^N # this doesn't matter much with Adam?
        train_best_params(hyper_model, params)

def grid_search_MLP_VAE():
    global hyper_model, max_params
    hyper_model = "MLPVAE_Incremental"
    reset_hyper_training(hyper_model)
    max_params = 20_000_000
    samples, _ = generate_training_data(None, hyper_stfts)  # full data-set, this may be more representative
    for vae_depth in range(2, 5):
        for latent in range(12, 4, -1):
            for ratio in [0.1, 0.5, 1.0, 2.0, 10.0]:
                params = [4, -5, latent, vae_depth, ratio]
                print(f"\n\n\nGrid Hyper-train: latent={latent}, vae_depth={vae_depth}")
                evaluate_model(params)

    train_topN_hyper_params()


def grid_search_Conv2D_AE():
    global hyper_model, max_params
    hyper_model = "Conv2D_AE"
    reset_hyper_training(hyper_model)
    max_params = 20_000_000
    samples, _ = generate_training_data(None, hyper_stfts)  # full data-set, this may be more representative
    for kernels in range(20, 9, -1):
        for layers in [7, 6, 5]:
            for size in range(10, 3, -1):
                params = [3, -6, layers, kernels, size]
                print(f"\n\n\nGrid Hyper-train: layers={layers}, kernels={kernels}, size={size}")
                evaluate_model(params)

    train_topN_hyper_params()

def grid_search_AudioConv_AE():
    global hyper_model, max_params, max_hyper_runs, max_loss
    k_size = int(sample_rate / middleCHz)
    hyper_model = "AudioConv_AE"
    reset_hyper_training(hyper_model)
    set_fail_loss(1_000_000)
    max_params = 1_000_000
    max_loss = 5 * audio_length
    samples, _ = generate_training_data(200, hyper_stfts)
    max_hyper_runs = 5 * 4 * 6 * 5
    count = 0
    for depth in [1, 2, 3, 4, 5]:
        for kernels in [10, 20, 40, 60]:
            for outer_kernel in exponential_interpolation(10, k_size, 6):
                outer_kernel = int(outer_kernel)
                for inner_kernel in exponential_interpolation(3, 20, 5):
                    inner_kernel = int(inner_kernel)

                    if depth == 1 and inner_kernel > 3:
                        break

                    params = [4, -6, depth, kernels, outer_kernel, inner_kernel]
                    count += 1
                    print(f"\n\n\nGrid Hyper-train #{count}/{max_hyper_runs}: layers={depth}, kernels={kernels}, outer={outer_kernel}, inner={inner_kernel}")
                    evaluate_model(params)

    train_topN_hyper_params()


if __name__ == '__main__':
    # Edit this to perform whatever operation is required.

    ###############################################################################################
    # MLP VAE model
    #full_hypertrain("StepWiseMLP")
    #full_hypertrain("MLPVAE_Incremental")

    #train_best_params("StepWiseMLP", [3, -5, 35, 3, 1.0]) # small batches converge faster!!
    #fine_tune("StepWiseMLP")
    #train_best_params("MLPVAE_Incremental", [5, -7, 6, 3, 10])
    #fine_tune("MLPVAE_Incremental")

    #grid_search_MLP_VAE()
    # train_best_params("StepWiseMLP")
    # train_best_params("MLPVAE_Incremental")
    # fine_tune("MLPVAE_Incremental")

    ###############################################################################################
    # RNN VAE model
    #full_hypertrain("RNNAutoEncoder")
    #full_hypertrain("RNN_VAE_Incremental")

    #train_best_params("RNNAutoEncoder")
    #train_best_params("RNN_VAE_Incremental")


    ###############################################################################################
    # 2D Convolution

    #full_hypertrain("Conv2D_AE")
    #grid_search_Conv2D_AE()
    #train_best_params("Conv2D_AE")
    #fine_tune("Conv2D_AE") # didn't achieve anything!

    #full_hypertrain("Conv2D_VAE_Incremental")
    #grid_search_Conv2D_VAE()
    #train_best_params("Conv2D_VAE_Incremental", [4, -5,  7, 4, 0.25])
    #fine_tune("Conv2D_VAE_Incremental")

    ###############################################################################################
    # Audio Convolution Auto-Encoder
    reset_hyper_training("AudioConv_AE")
    grid_search_AudioConv_AE()

    #
    # #full_hypertrain("AudioConv_AE")
    # optimise_hyper_parameters("AudioConv_AE")
    # train_best_params("AudioConv_AE")
