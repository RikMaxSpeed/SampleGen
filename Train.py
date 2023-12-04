import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from IPython.display import FileLink
import math

from AutoEncoderModels import *
from MakeSTFTs import *
from Graph import *
from ModelUtils import *
from Augment import *




def log_interp(start, end, steps):
    return torch.exp(torch.linspace(math.log(start), math.log(end), steps))




def predict_stft(model, input_stft, randomise):
    input_stft = convert_stft_to_input(input_stft)

    # Add an extra dimension for batch (if not already present)
    if len(input_stft.shape) == 2:
        input_stft = input_stft.unsqueeze(0)
    
    input_stft = input_stft.to(device)
    
    with torch.no_grad():
        loss, predicted_stft = model.forward_loss(input_stft)

    predicted_stft = predicted_stft.squeeze(0)
    return convert_stft_to_output(predicted_stft), loss


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
    assert(stfts.shape[1] == stft_buckets)
    assert(stfts.shape[2] == sequence_length)


# Hyper-parameter optimisation
max_loss = 1000 # large value to tel the hyper-parameter optimiser not to go here.
last_saved_loss = 0.02 # don't bother saving models above this threshold

# Keep track of all the test-losses over multiple runs, so we can learn how to terminate early on poor hyper-parameters.
all_test_losses = []
all_test_names = []


    
    
# Main entry point for training the model
def train_model(hyper_params, max_time, max_params, max_overfit, verbose):
    print(f"train_model: hyper-parameters={hyper_params}")
    
    # We split the hyper-params into optimiser parameters & model parameters:
    opt_params   = hyper_params[0:3]
    model_params = hyper_params[3:]
    print(f"opt_params={opt_params}, model_params={model_params}")
    
    # Optmiser parameters:
    batch_size, learning_rate, weight_decay = opt_params
    batch_size = int(batch_size) # required even though it's declared integer in the search-space :(
    optimiser_text = f"batch={batch_size}, learning_rate={learning_rate:.1g}, weight_decay={weight_decay:.1g}"
    print(f"optimiser: {optimiser_text}")
    
    # Create the model
    model, model_text = make_model(model_params, max_params, verbose)
    if model is None:
        return max_loss
    print(f"model: {model_text}")
    
    # Train/Test & DataLoader
    dataset = TensorDataset(stfts)
    train_dataset, test_dataset = split_dataset(dataset, 0.8)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"train={len(train_dataset)} samples, batch={batch_size} --> {len(train_dataset)/batch_size:.1f} batches/epoch")

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    start = time.time()
    lastGraph = start
    train_losses = []
    test_losses = []
    
    # Stopping condition
    window     = 15 # check progress between two windows
    min_change = 0.005 # stop if lossNew/lossOld - 1 < min_change

    graph_interval = 5
    
    max_epochs = 10000 # currently irrelevant
    
    for epoch in range(1, max_epochs):
        model.train() # ensure we compute gradients
        
        for batch_idx, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
        
            # Forward pass
            loss, _ = model.forward_loss(inputs)
            
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
            
        
        # Progress
        now = time.time()
        total = now - start


        # Save the best models (but not too often)
        global last_saved_loss
        if epoch > 20 and train_losses[-1] < last_saved_loss * 0.95:
            last_saved_loss = train_losses[-1]
            filename = model_text # keep over-writing the same file as the loss improves
            print("*** Best! loss={:.4f}, model={}, hyper={}".format(last_saved_loss, model_text, optimiser_text))
            torch.save(model.state_dict(), filename + ".wab")
            with open(filename+".txt", 'w') as file:
                file.write(model_text + "\n")
                file.write(f"optimiser: {optimiser_text}\n")
                file.write("\n")
                file.write(f"time={total:.0f} sec, train_size={len(train_dataset)}, batch_size={batch_size}, epoch={epoch} = {total/epoch:.1f} sec/epoch\n")
                file.write(f"\ttrain loss={train_losses[-1]:.5f}\n")
                file.write(f"\ttest  loss={test_losses[-1]:.5f}\n")
                file.write("\n")
                file.write(str(hyper_params))


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
        # Note: this should really be time-based rather than epoch as models run at different speeds depending on batch size, learning rate and model size.
        # ie: if after 1mn the model is worse than the average at 1mn then give up...
        if epoch > 40 and epoch % 25 == 0:
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
