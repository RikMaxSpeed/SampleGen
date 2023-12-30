# Use a GPR to adjust the hpyer-parameters.
# The best models are saved to disk.
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from MakeSTFTs import audio_length
import time

max_params = 0
tuning_count = 0
break_on_exceptions = False # True=Debugging, False allows the GPR to continue even if the model blows up (useful for long tuning runs!)
max_loss = audio_length # default

hyper_model = "None"
hyper_losses = []
hyper_names  = []
hyper_params = []
max_hyper_runs = 80  # it usually gets stuck at some local minimum well before this.
hyper_stfts = False
hyper_start = time.time()

def reset_hyper_training(model_name):
    global hyper_model, hyper_stfts, hyper_losses, hyper_names, hyper_params
    hyper_model = model_name
    hyper_stfts = model_uses_STFTs(model_name)
    hyper_losses = []
    hyper_names = []
    hyper_params = []
    reset_train_losses(model_name, True)
    hyper_start = time.time()

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
    
    max_overfit = 1.3 # Ensure we retain the models that generalise reasonably well.
    
    max_epochs = 80 # This is sufficient to figure out which model will converge best if we let it run for longer.
    if is_incremental_vae(hyper_model) or is_audio(hyper_model): # fast models
        max_overfit = 2
        max_epochs = 5000 # assuming we'll bump into max_time first!

    max_time = 3*60 # we don't like slow models...
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

    elapsed = time.time() - hyper_start
    print(f"Hyper-tuning: total {count} iterations in time={int(elapsed):,} sec = {int(elapsed/count):,} sec/iteration")

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
    how_many = None # full data-set, this may be more representative
    if model_name == "AudioConv_AE":
        how_many = 200 # 80% = 10 x batch=16
        # In practice, this is very difficult data set to converge on!

    samples, _ = generate_training_data(how_many, hyper_stfts)

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
            search_space.append(Integer(3,        6,   'uniform',       name='layer_count'))
            search_space.append(Integer(1,       30,   'uniform',       name='kernel_count'))
            search_space.append(Integer(2,       10,   'uniform',       name='kernel_size'))

        case "Conv2D_VAE_Incremental":
            search_space.append(Integer(5,        20,   'uniform',      name='latent_size'))
            search_space.append(Integer(2,         5,   'uniform',      name='vae_depth'))
            search_space.append(Real   (0.1,      10,   'uniform',      name='vae_ratio'))

        case "AudioConv_AE":
            # audio_length, depth, kernel_count, kernel_size, stride
            max_kernel_size = int(sample_rate / middleCHz)
            search_space.append(Integer( 2,     7,     'uniform',  name='layers'))
            search_space.append(Integer(20,    80,     'log-uniform',  name='kernels'))
            search_space.append(Integer(20,    max_kernel_size, 'log-uniform',  name='kernel_size'))
            search_space.append(Integer( 30,    80, 'log-uniform',  name='compression'))

        case "AudioConv_VAE_Incremental":
            search_space.append(Integer(5, 20, 'uniform', name='VAE latent'))
            search_space.append(Integer(2, 5, 'uniform', name='depth'))
            search_space.append(Real(0.1, 10, 'uniform', name='ratio'))

        case "AudioConv_VAE":
            search_space.append(Integer(   3,     5,    'uniform',  name='layers'))
            search_space.append(Integer(  25,    60,    'uniform',  name='kernels'))
            search_space.append(Integer(  50,   160,'log-uniform',  name='size'))
            search_space.append(Integer( 0.5,     8,'log-uniform',  name='ratio'))
            search_space.append(Integer(5, 20, 'uniform', name='VAE latent'))
            search_space.append(Integer(2, 5, 'uniform', name='depth'))
            search_space.append(Real(0.1, 10, 'uniform', name='ratio'))

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
    train_best_params(model_name, params, finest=True)


def train_best_params(model_name, params = None, finest = False):
    if finest:
        print(f"\n\n\nFine-Tuning {model_name}\n")
    else:
        print(f"\n\n\nTraining model {model_name}\n")

    reset_train_losses(model_name, False)

    #generate_training_data(200, hyper_stfts) # Small dataset of the most difficult samples
    generate_training_data(None, hyper_stfts) # Full dataset with no augmentation
    #generate_training_stfts(3000) # use a large number of samples with augmentation

    if params is None:
        model_name, params, _ = get_best_configuration_for_model(model_name)

    print(f"train_best_params: {model_name}: {params}")
    start_new_stft_video(f"STFT - train {model_name}", True)

    max_time = 2 * hour # hopefully the model converges way before this!
    max_overfit = 2.0 # if we set this too high, the VAE stdevs become huge and the model doesn't generalise.

    # if "VAE" in model_name: # Allow the Auto-Encoders to over-fit
    #     max_overfit = 10.0
    #     print("Overriding max_overfit={max_overfit}\n")

    max_params = 1e9  # not relevant, we have a valid model
    max_epochs = 9999 # ignore
    max_loss = audio_length
    set_fail_loss(audio_length)

     # This does improve the final accuracy, but it's very slow.
    if finest:
        params[0] =  2 # override the batch-size
        params[1] = -7 # override the learning rate
        #max_time = hour

    #set_display_hiddens(True) # Displays the internal auto-encoder output

    verbose = True
    train_model(model_name, params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, finest)


def full_hypertrain(model_name):
    optimise_hyper_parameters(model_name)
    train_best_params(model_name)
    fine_tune(model_name)


def train_topN_hyper_params(topN = 5):
    # Train the top N but with longer time spans
    say_out_loud("Training top hyper parameters!")
    reset_train_losses(hyper_model, True)
    topN = min(topN, len(hyper_losses))
    order = np.argsort(hyper_losses)
    for i in range(topN):
        n = order[i]
        print(f"\n\n\n\nTraining optimised parameters rank #{i+1}: {hyper_names[n]}")
        params = hyper_params[n]
        params[0] = 2  # batch-size = 2^N
        params[1] = -7 # learning rate = 10^N # this doesn't matter much with Adam?
        train_best_params(hyper_model, params)



def grid_search(model_name, param_values, data_size=None, max_model_size=30_000_000):
    runs = 0
    print(f"Grid search for {model_name}:")
    for i in range(len(param_values)):
        print(f"\tparameter#{i+1}: values={param_values[i]}")
    print("\n")

    def recurse_through_parameter_values(params_so_far, remaining_param_values):
        if len(remaining_param_values) == 0:
            nonlocal runs
            global max_hyper_runs
            runs += 1
            print(f"\n\nGrid Hyper-train #{runs}/{max_hyper_runs}: {params_so_far}")
            evaluate_model(params_so_far)
            return

        next_remain = remaining_param_values[1:]
        for value in remaining_param_values[0]:
            next_params = params_so_far + [value]
            recurse_through_parameter_values(next_params, next_remain)


    global hyper_model, max_params, max_hyper_runs
    hyper_model = model_name
    reset_hyper_training(hyper_model)
    max_params = max_model_size

    samples, _ = generate_training_data(data_size, hyper_stfts)

    max_hyper_runs = 1
    for values in param_values:
        max_hyper_runs *= len(values)

    opt_params = [4, -6] # batch-size and learning rate

    start = time.time()

    recurse_through_parameter_values(opt_params, param_values)

    print("\n\nGrid Hyper-training complete!")

    elapsed = time.time() - start
    elapsed_minutes = elapsed // 60
    print(f"Grid Hyper-training took {elapsed_minutes:,} minutes for {runs} iterations = {elapsed/runs:.1f} sec/iteration\n\n")

    train_topN_hyper_params()


def grid_search_MLP_VAE():
    grid_search("MLPVAE_Incremental",
                [list(range(2, 5)),
                 list(range(12, 4, -1)),
                 [0.1, 0.5, 1.0, 2.0, 10.0]])

def grid_search_Conv2D_VAE():
    grid_search("Conv2D_VAE_Incremental",
                [list(range(6, 10)), # latent
                 [3, 4], # layers
                 exponential_interpolation(0.1, 0.5, 5) # ratio
                 ])


def grid_search_Conv2D_AE():
    grid_search("Conv2D_AE",
                [list([7, 6, 5]),  # layers
                 list(range(20, 9, -1)), # kernels
                 list(range(10, 3, -1)) # size
                 ])

def grid_search_AudioConv_AE():
    max_params = 5_000_000
    data_size = 200
    max_kernel_size = int(sample_rate / middleCHz)

    grid_search("AudioConv_AE",
                [[4, 3, 2], # [5, 4, 3, 2] # depth
                 [30], #exponential_interpolation(25, 35, 2, True), # kernel count
                 exponential_interpolation(max_kernel_size//16, max_kernel_size, 10, True), # kernel size
                 #exponential_interpolation(max_kernel_size / 2, max_kernel_size, 2, True),  # kernel size
                 [80] # compression
                 ],
                data_size,
                max_params)


def grid_search_AudioConv_VAE_I():
    grid_search("AudioConv_VAE_Incremental",
                [
                    [10, 9, 8, 7, 6], #[exponential_interpolation(10, 6, 5, True), # latent
                    [4, 3], # layers
                    exponential_interpolation(1.0, 0.1, 6) # ratio
                ])

def hypertrain_MLP_VAE():
    ###############################################################################################
    # MLP VAE model
    full_hypertrain("StepWiseMLP")
    full_hypertrain("MLPVAE_Incremental")

    #train_best_params("StepWiseMLP", [3, -5, 35, 3, 1.0]) # small batches converge faster!!
    #fine_tune("StepWiseMLP")
    #train_best_params("MLPVAE_Incremental", [5, -7, 6, 3, 10])
    #fine_tune("MLPVAE_Incremental")

    #grid_search_MLP_VAE()
    # train_best_params("StepWiseMLP")
    # train_best_params("MLPVAE_Incremental")
    # fine_tune("MLPVAE_Incremental")

def hypertrain_RNN_VAE():
    ###############################################################################################
    # RNN VAE model
    full_hypertrain("RNNAutoEncoder")
    full_hypertrain("RNN_VAE_Incremental")

    #train_best_params("RNNAutoEncoder")
    #train_best_params("RNN_VAE_Incremental")


def hypertrain_Conv2D_VAE():
    ###############################################################################################
    # 2D Convolution

    #full_hypertrain("Conv2D_AE")
    #grid_search_Conv2D_AE()
    #train_best_params("Conv2D_AE")
    #fine_tune("Conv2D_AE") # didn't achieve anything!

    #full_hypertrain("Conv2D_VAE_Incremental")
    #grid_search_Conv2D_VAE()
    train_best_params("Conv2D_VAE_Incremental", [4, -5,  7, 4, 0.25])
    #fine_tune("Conv2D_VAE_Incremental")


def hypertrain_AudioConv_VAE():
    ###############################################################################################
    # Audio Convolution Auto-Encoder

    set_fail_loss(20_000)
    #full_hypertrain("AudioConv_AE") # MPS crashes after a while :(
    train_best_params("AudioConv_AE", [2, -4, 4, 40, 34, 30])
    full_hypertrain("AudioConv_VAE_Incremental")

    #grid_search_AudioConv_AE()
    #grid_search_AudioConv_VAE_I()

    #train_best_params("AudioConv_AE", [4, -6, 3, 30, 186, 30])

    #train_best_params("AudioConv_VAE_Incremental", [4, -6, 20, 3, 0.1])
    #train_best_params("AudioConv_VAE_Incremental", [4, -6, 8, 3, 0.1])

    # full_hypertrain("AudioConv_VAE")

from Generate import Sample_Generator, g, use_model

if __name__ == '__main__':
    hypertrain_AudioConv_VAE()

    # g = Sample_Generator("AudioConv_VAE_Incremental")
    # g.speed_test()

