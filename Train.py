import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from IPython.display import FileLink
import math

from MakeModels import *
from MakeSTFTs import *
from Graph import *
from ModelUtils import *
from Augment import *




def log_interp(start, end, steps):
    return torch.exp(torch.linspace(math.log(start), math.log(end), steps))




def predict_stft(model, input_stft):
    input_stft = convert_stft_to_input(input_stft)

    # Add an extra dimension for batch (if not already present)
    if len(input_stft.shape) == 2:
        input_stft = input_stft.unsqueeze(0)
    
    input_stft = input_stft.to(device)
    
    with torch.no_grad():
        loss, predicted_stft = model.forward_loss(input_stft)

    predicted_stft = predicted_stft.squeeze(0)
    return convert_stft_to_output(predicted_stft), loss.item()


# Sample data
train_dataset    = None
test_dataset     = None
sanity_test_stft = None
sanity_test_name = None


def generate_training_stfts(how_many):
    global sanity_test_stft, sanity_test_name, train_dataset, test_dataset
                
     # Augmentation is used if this exceeds the number of real available samples
    stfts, file_names = get_training_stfts(how_many)

    lengths = np.array([x.shape[1] for x in stfts])
    plot_multiple_histograms_vs_gaussian([lengths * stft_hop / sample_rate], ["Sample Durations (seconds)"])

    # Pick an example file to sanity check that everything is behaving from A-Z
    for i in range(len(file_names)):
        if "grand piano c3" in file_names[i].lower():
            sanity_test_stft = stfts[i]
            sanity_test_name = file_names[i]
            break

    stfts = convert_stfts_to_inputs(stfts)
    train_stfts, test_stfts = split_dataset(stfts, 0.8)
    
    # Training set is kept completely separate from Test when augmenting.
    if how_many is not None and len(train_stfts) < how_many:
        train_stfts = augment_stfts(train_stfts, how_many)
    
    train_dataset = train_stfts
    test_dataset  = test_stfts
    
    print(f"Using train={len(train_dataset)} samples, test={len(test_dataset)} samples.")
    
    


# Hyper-parameter optimisation
max_loss = 100 # large value to tel the hyper-parameter optimiser not to go here.
last_saved_loss = 0.02 # don't bother saving models above this threshold

# Keep track of all the test-losses over multiple runs, so we can learn how to terminate early on poor hyper-parameters.
all_test_losses = []
all_test_names = []

# Compare the training loss to the best we've found, and abort if it's too far off.
best_train_losses = []



# Main entry point for training the model
def train_model(hyper_params, max_epochs, max_time, max_params, max_overfit, verbose):
    
    # We split the hyper-params into optimiser parameters & model parameters:
    opt_params   = hyper_params[0:3]
    model_params = hyper_params[3:]
    
    # Optmiser parameters:
    batch_size, learning_rate, weight_decay = opt_params
    batch_size = int(batch_size) # convert int64 to int32
    optimiser_text = f"batch={batch_size}, learning_rate={learning_rate:.1g}, weight_decay={weight_decay:.1g}"
    print(f"optimiser: {optimiser_text}")
    
    # Create the model
    model, model_text = make_model(model_params, max_params, verbose)
    if model is None:
        return max_loss
    print(f"model: {model_text}")
    
    # Train/Test & DataLoader
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
    window     = 10 # check progress between two windows
    min_change = 0.005 # stop if lossNew/lossOld - 1 < min_change

    graph_interval = 5
    
    for epoch in range(1, max_epochs):
        model.train() # ensure we compute gradients
        
        #for batch_idx, (inputs,) in enumerate(dataloader):
        for batch_idx, inputs in enumerate(dataloader):
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
        global last_saved_loss, best_train_losses
        if epoch >= 10 and train_losses[-1] < last_saved_loss * 0.95:
            last_saved_loss = train_losses[-1]
            
            # Save the model:
            file_name = model_text # keep over-writing the same file as the loss improves
            print("*** Best! loss={:.4f}, model={}, optimiser={}".format(last_saved_loss, model_text, optimiser_text))
            print(f"hyper-parameters: [hyper_params]")
            torch.save(model.state_dict(), file_name + ".wab")
            
            # Write the parameters to file:
            with open(file_name+".txt", 'w') as file:
                file.write(model_text + "\n")
                file.write(f"optimiser: {optimiser_text}\n")
                file.write("\n")
                file.write(f"time={total:.0f} sec, train_size={len(train_dataset)}, batch_size={batch_size}, epoch={epoch} = {total/epoch:.1f} sec/epoch\n")
                file.write(f"\ttrain loss={train_losses[-1]:.5f}\n")
                file.write(f"\ttest  loss={test_losses[-1]:.5f}\n")
                file.write("\n")
                file.write(str(hyper_params))
            
            # Generate a test tone:
            resynth, loss = predict_stft(model, sanity_test_stft)
            save_and_play_audio_from_stft(resynth, sample_rate, stft_hop, f"Results/{sanity_test_name} {model_text} - resynth.wav", False)
            

        if verbose and now - lastGraph > graph_interval:
            plot_losses(train_losses, test_losses)
            lastGraph = now
            graph_interval = int(min(3600, 1.5*graph_interval))


        if stop_condition(train_losses, test_losses, window, min_change, max_overfit, total, epoch):
            print("Training is stalled...")
            break
    
        if epoch < 5: # Test a random sample to show that the code is working from A-Z
            resynth, loss = predict_stft(model, sanity_test_stft)

        if total > max_time:
            print("Total time={:.1f} exceeds max={:.0f}sec".format(total, max_time))
            break
            
        # Early stopping based on average convergence:
        # Note: this should really be time-based rather than epoch as models run at different speeds depending on batch size, learning rate and model size.
        # ie: if after 1mn the model is worse than the average at 1mn then give up...
        # In practice, the current implementation doesn't work too well, the stdev can be very high. Could try mean - 0.1 x stdev ? ...
#        if epoch > 30 and epoch % 10 == 0:
#            mean, stdev = compute_epoch_stats(all_test_losses, epoch, 10)
#            loss = test_losses[-1]
#            if mean is not None and loss > mean: # we could make this more aggressive, for example: mean - 0.5 * stdev
#                print(f"Early stopping at epoch={epoch}, test loss={loss:.5f} vs mean={mean:.5f}")
#                break

        if epoch > 20 and epoch < len(best_train_losses) and train_loss[epoch] < best_train_losses[epoch] * 1.5:
            print(f"Early stopping at epoch={epoch}, train loss={train_loss[epoch]:.5f} vs best={best_train_losses[epoch]:.5f}")
            break
            

    # Done!
    if len(best_train_losses) == 0 or np.min(train_losses) < np.min(best_train_losses):
        best_train_losses = train_losses
    
    trainL  = train_losses[-1]
    testL   = test_losses[-1]
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




