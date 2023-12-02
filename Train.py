import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from IPython.display import FileLink
import math

from AutoEncoder import *
from MakeSTFTs import *
from Graph import *
from ModelUtils import *
from Augment import *


sample_duration = 2 # seconds
sequence_length = int(sample_duration * sample_rate / stft_hop)
input_size = stft_buckets * sequence_length



def get_best_hyper_params():
    return [6, 7.753192086063947, 7.411389428825689, 5.301401097330652, 9, 0.0009685451968313163, 5.151727534054279e-05]
    #return [8, 3.7266841851402606, 3.787922442988532, 8.189781767196333, 9, 4.5017602108986394e-05, 1.767746772905285e-07]
    
def get_best_filename():
    return "Model latent=6, layer3=46, layer2=340, layer1=1802, loss=0.0063.wab"
    #return "Model latent=8, layer3=29, layer2=109, layer1=892, loss=0.0026.wab" # Mu=1 (linear, no transform), small latent size


def exponential_interpolation(start, end, N):
    return [start * (end/start) ** (i/(N-1)) for i in range(N)]

    
    
def adjust_stft_length(stft_tensor, target_length):
    
    assert(stft_buckets <= stft_tensor.shape[0] <= stft_buckets + 1)
    current_length = stft_tensor.shape[1]
    
    # If the current length is equal to the target, return the tensor as is
    if current_length == target_length:
        return stft_tensor

    # If the current length is greater than the target, truncate it
    if current_length > target_length:
        return stft_tensor[:, :target_length]

    # If the current length is less than the target, pad it with zeros
    padding_length = target_length - current_length
    padding = torch.zeros((stft_tensor.shape[0], padding_length))
    
    return torch.cat((stft_tensor, padding), dim=1)


def transpose(tensor):
    return tensor.transpose(0, 1) # Swap the SFTF bins and sequence length


def convert_stft_to_input(stft):
    assert(len(stft.shape) == 2)
    assert(stft.shape[0] == stft_buckets + 1)
    stft = stft[1:, :]
    assert(stft.shape[0] == stft_buckets)
    
    stft = adjust_stft_length(stft, sequence_length)
    stft = complex_to_mulaw(stft)
    stft = transpose(stft)
    
    return stft.contiguous()


def convert_stft_to_output(stft):
    #debug("convert_stft_to_output.stft", stft)
    
    # Fix dimensions
    amplitudes = stft.squeeze(0)
    amplitudes = transpose(amplitudes)
    
    # Get rid of any silly values...
    amplitudes[amplitudes <= 0] = 0
    
    # Add back the constant bucket = 0
    assert(amplitudes.shape[0] == stft_buckets)
    zeros = torch.zeros(amplitudes.size(1)).unsqueeze(0).to(device)
    amplitudes = torch.cat((zeros, amplitudes), dim=0)
    assert(amplitudes.shape[0] == stft_buckets + 1)
    
    # Convert back to complex with 0 phase
    output_stft = mulaw_to_zero_phase_complex(amplitudes)
    assert(output_stft.shape[0] == stft_buckets + 1)
    output_stft = output_stft.cpu().detach().numpy()

    return output_stft
        

def test_stft_conversions(filename):
    sr, stft = compute_stft_for_file(filename, 2*stft_buckets, stft_hop)
    debug("stft", stft)
    
    if False: # Generate a synthetic spectrum
        for f in range(stft.shape[0]):
            for t in range(stft.shape[1]):
                stft[f, t] = 75*np.sin(f*t) if f>=2*t and f <= 3*t else 0
    
    plot_stft(filename, stft, sr, stft_hop)
    stft = np.abs(stft) # because we discard the phases, everything becomes positive amplitudes
    
    tensor = torch.tensor(stft)
    input = convert_stft_to_input(tensor).to(device)
    output = convert_stft_to_output(input)
    
    plot_stft("Resynth " + filename, output, sr, stft_hop)
    save_and_play_audio_from_stft(output, sr, stft_hop, "Results/resynth-mulaw-" + os.path.basename(filename), True)

    diff = np.abs(output - stft)
    debug("diff", diff)
    plot_stft("Diff", diff, sr, stft_hop)
    for f in range(stft.shape[0]):
        for t in range(stft.shape[1]):
            d = diff[f, t]
            if d > 0.1:
                print("f={}, t={}, diff={:.3f}, stft={:.3f}, output={:.3f}".format(f, t, d, stft[f, t], output[f, t]))


#test_stft_conversions("Samples/Piano C4 Major 13.wav")
#test_stft_conversions("/Users/Richard/Coding/WaveFiles/FreeWaveSamples/Alesis-S4-Plus-Clean-Gtr-C4.wav")
#sys.exit(1)


def log_interp(start, end, steps):
    return torch.exp(torch.linspace(math.log(start), math.log(end), steps))




def predict_stft(model, input_stft, randomise):
    input_stft = convert_stft_to_input(input_stft)

    # Add an extra dimension for batch (if not already present)
    if len(input_stft.shape) == 2:
        input_stft = input_stft.unsqueeze(0)
    
    input_stft = input_stft.to(device)
    
    with torch.no_grad():
        predicted_stft, mu, logvar = model.forward(input_stft, randomise)
    
    loss = model.loss_function(predicted_stft, input_stft, mu, logvar).item()

    return convert_stft_to_output(predicted_stft), loss


def mlp_size(layer_sizes):
    total_params = 0
    
    for i in range(len(layer_sizes) - 1):
        total_params += (layer_sizes[i] * layer_sizes[i+1]) + layer_sizes[i+1]
        
    return total_params


# Sample data
stfts = []
file_names = []
count = -1
sanity_test_stft = None

def generate_training_stfts(how_many):
    global stfts, filenames, count, sanity_test_stft
    
    if how_many == count:
        return # no need to do this again
        
    count = how_many
    
     # Augmentation is used if this exceeds the number of real available samples
    stfts, file_names = get_training_stfts(count)
    assert(len(stfts) == count)
    
    print(f"Using {count} STFTs")
    sanity_test_stft = stfts[7] # Used to prove everything's working from A-Z
    lengths = np.array([x.shape[1] for x in stfts])
    plot_multiple_histograms_vs_gaussian([lengths * stft_hop / sample_rate], ["Sample Durations (seconds)"])
    stfts = torch.stack([convert_stft_to_input(stft) for stft in stfts])
    print("Input STFTs: {} x {}".format(stfts.shape, stfts.dtype))
    assert(stfts.shape[0] == count)
    assert(stfts.shape[1] == sequence_length)
    assert(stfts.shape[2] == stft_buckets)


# Hyper-parameter optimisation
max_loss = 1000 # large value to tel the hyper-parameter optimiser not to go here.
last_saved_loss = 0.02 # don't bother saving models above this threshold

# Keep track of all the test-losses over multiple runs, so we can learn how to terminate early on poor hyper-parameters.
all_test_losses = []
all_test_names = []


    
    
# Main entry point for training the model
def train_model(hyper_params, max_time, max_ratio, max_overfit, verbose):
    print("train_model: hyper-parameters={}".format(hyper_params))

    # Extract and display hyper-parameters:
    latent_size, layer3_ratio, layer2_ratio, layer1_ratio, batch_size, learning_rate, weight_decay = hyper_params
    layers = get_layers(hyper_params)
    batch_size = int(batch_size) # required even though it's declared integer in the search-space :(
    
    # Check this model isn't too large (without allocating any memory!)
    train = stfts.numel()
    size = 2 * mlp_size(layers) # encode and decode, this is pretty accurate, it's missing the stdev layer + ?
    print(f"layers={layers} -> approx model size={size:,} parameters")
    if size > train * max_ratio * 1.1:
        print("Aborting: max model size exceeded.")
        return max_loss + size*1e-5

    
    # Output some pretty text:
    model_text = f"latent={layers[4]}, layer3={layers[3]}, layer2={layers[2]}, layer1={layers[1]}"
    optimiser_text = f"batch={batch_size}, learning_rate={learning_rate:.1g}, weight_decay={weight_decay:.1g}"
    print("hyper-parameters:\n\tmodel: " + model_text + "\n\toptimiser: " + optimiser_text)
    
    # Now we've checked it doesn't have a silly size, we can actually instantiate the model with these parameters
    model = make_model(hyper_params, verbose)
    
    # Check the actual model size:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"model parameters={params:,} vs approx={size:,}, ratio={params/size:.2f}") # the approx is well within 1%
    print("{:,} trainable parameters vs {:,} inputs, ratio={:.1f} to 1".format(params, train, params/train))
    if params/train > max_ratio:
        print("Aborting: model is too large.")
        return max_loss + params/train
        
    # Train/Test & DataLoader
    dataset = TensorDataset(stfts)
    train_dataset, test_dataset = split_dataset(dataset, 0.8)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"train={len(train_dataset)}, batch={batch_size} --> {len(train_dataset)/batch_size:.1f} batches/epoch")

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    start = time.time()
    lastGraph = start
    train_losses = []
    test_losses = []
    
    # Stopping condition
    window = 20 # check progress between two windows
    min_change = 0.005 # stop if lossNew/lossOld - 1 < min_change

    graph_interval = 5
    
    max_epochs = 10000 # currently irrelevant
    
    for epoch in range(1, max_epochs):
        model.train() # ensure we compute gradients
        
        for batch_idx, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
        
            # Forward pass
            outputs, mus, logvars = model.forward(inputs, True)
            loss = model.loss_function(outputs, inputs, mus, logvars)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                
        # After each epoch
        train_losses.append(compute_average_loss(model, train_dataset, batch_size))
        test_losses.append(compute_average_loss(model, test_dataset, batch_size))
        if np.isnan(train_losses[-1]) or np.isnan(test_losses[-1]):
            print("Aborting: model returns NaNs :(") # Happens when the learning rate is too high
            return max_loss
            
        # Save the best models (but not too often)
        global last_saved_loss
        if epoch > 20 and train_losses[-1] < last_saved_loss * 0.95:
            last_saved_loss = train_losses[-1]
            filename = f"Model {model_text}, loss={last_saved_loss:.4f}.wab"
            print("*** Best! loss={:.4f}, model={}, hyper={}".format(last_saved_loss, model_text, optimiser_text))
            torch.save(model.state_dict(), filename)
        
        # Progress
        now = time.time()
        total = now - start
        
        if verbose and now - lastGraph > graph_interval:
            plot_losses(train_losses, test_losses)
            lastGraph = now
            graph_interval = int(min(3600, 1.5*graph_interval))

        if stop_condition(train_losses, test_losses, window, min_change, max_overfit, total, epoch):
            print("Training is stalled...")
            break
    
        if epoch < 5: # Test a random sample to show that the code is working from A-Z
            resynth, loss = predict_stft(model, sanity_test_stft, True)

        if total > max_time:
            print("Total time={:.1f} exceeds max={:.0f}sec".format(total, max_time))
            break
            
        # Early stopping based on average convergence:
        if epoch % 25 == 0:
            mean, stdev = compute_epoch_stats(all_test_losses, epoch, 10)
            loss = test_losses[-1]
            if mean is not None and loss > mean: # we could make this more aggressive, for example: mean - 0.5 * stdev
                print(f"Early stopping at epoch={epoch}, test loss={loss:.5f} vs mean={mean:.5f}")
                break

    # Done!
    trainL  = train_losses[-1]
    testL   =  test_losses[-1]
    elapsed = time.time() - start
    epochs  = len(train_losses)
    print("\n\nFinished Training after {} epochs in {:.1f} sec ({:.2f} sec/epoch), sample duration={:.1f} sec, test loss={:.2f}, train loss={:.2f}, overfit={:.1f}"\
    .format(epochs, elapsed, elapsed/epochs, sample_duration, testL, trainL, testL/trainL))
    
    all_test_losses.append(test_losses)
    all_test_names.append("loss={:.5f}, {}, {}".format(np.min(test_losses), model_text, optimiser_text))
    plot_multiple_losses(all_test_losses, all_test_names, 5) # this could become large...
    
    if verbose:
        plot_losses(train_losses, test_losses)
    
    return np.min(test_losses)



def get_layers(hyper_params):
    latent_size, layer3_ratio, layer2_ratio, layer1_ratio, batch_size, learning_rate, weight_decay = hyper_params

    layer3_size = int(latent_size * layer3_ratio)
    layer2_size = int(layer3_size * layer2_ratio)
    layer1_size = int(layer2_size * layer1_ratio)
    assert(latent_size <= layer3_size <= layer2_size <= layer1_size)
    
    layers = [stft_buckets * sequence_length, layer1_size, layer2_size, layer3_size, latent_size]

    return layers
    
    
def make_model(hyper_params, verbose):
    layers = get_layers(hyper_params)
    model = STFTVariationalAutoEncoder(sequence_length, stft_buckets, layers[1:], nn.ReLU())
    model.float() # ensure we're using float32 and not float64
    model.to(device)
    
    if verbose:
        print("model={}".format(model))
        
    return model


def load_best_model():
    model = make_model(get_best_hyper_params(), True)
    model.load_state_dict(torch.load(get_best_filename()))
    model.eval() # Ensure the model is in evaluation mode
    model.to(device)
    
    return model



from IPython.display import HTML, display


def display_custom_link(file_path, display_text=None):

    if display_text is None:
        display_text = file_path

    link_str = f'<a href="{file_path}" target="_blank">{display_text}</a>'
    display(HTML(link_str))    



def test_all():
    model = load_best_model()
        
    stfts, file_names = load_STFTs()
    
    names=[]
    losses=[]
    
    noisy = False
    graphs = False
    
    for i in range(len(stfts)):
        stft = stfts[i]
        name = file_names[i][:-4]
        
        stft = adjust_stft_length(stft, sequence_length)
        
        if graphs:
            plot_stft(name, stft, sample_rate)
        
        if noisy:
            save_and_play_audio_from_stft(stft.cpu().numpy(), sample_rate, stft_hop, None, True)
        
        resynth, loss = predict_stft(model, stft, False)
        names.append(name)
        losses.append(loss)
        
        if graphs:
            plot_stft("Resynth " + name, resynth, sample_rate)
        
        save_and_play_audio_from_stft(resynth, sample_rate, stft_hop, "Results/" + name + " - resynth.wav", noisy)
        
    indices = [i[0] for i in sorted(enumerate(losses), key=lambda x:x[1])]
    pad = max([len(x) for x in names])
    for i in indices:
        loss = losses[i]
        name = names[i]
        display_custom_link("Results/" + name + " - resynth.wav", "{}: loss={:.6f}".format(name, loss))
        
    plot_multiple_histograms_vs_gaussian([losses], ["Resynthesis Loss"])
