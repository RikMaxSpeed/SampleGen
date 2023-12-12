import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from IPython.display import FileLink
import math

from ModelUtils import *
from MakeModels import *
from MakeSTFTs import *
from Graph import *
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
    
    #display_average_stft(stfts, True)
    
    # Training set is kept completely separate from Test when augmenting.
    if how_many is not None and len(train_stfts) < how_many:
        train_stfts = augment_stfts(train_stfts, how_many)
    
    train_dataset = train_stfts
    test_dataset  = test_stfts
    
    print(f"Using train={len(train_dataset)} samples, test={len(test_dataset)} samples.")
    
    


# Hyper-parameter optimisation
last_saved_loss = 100 # don't bother saving models above this threshold

# Keep track of all the test-losses over multiple runs, so we can learn how to terminate early on poor hyper-parameters.
all_test_losses = []
all_test_names = []

# Compare the training loss to the best we've found, and abort if it's too far off.
best_train_losses = []


use_exact_train_loss = False # Setting to True is more accurate but very expensive in CPU time


# Main entry point for training the model
def train_model(model_type, hyper_params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose):
    
    # We split the hyper-params into optimiser parameters & model parameters:
    opt_params   = hyper_params[0:3]
    model_params = hyper_params[3:]
    
    # Optmiser parameters:
    batch, learning_rate, weight_decay = opt_params
    batch_size = int(2 ** batch) # convert int64 to int32
    learning_rate *= batch_size # see https://www.baeldung.com/cs/learning-rate-batch-size
    optimiser_text = f"Adam batch={batch_size}, learning_rate={learning_rate:.2g}, weight_decay={weight_decay:.2g}"
    print(f"optimiser: {optimiser_text}")
    
    # Create the model
    model, model_text = make_model(model_type, model_params, max_params, verbose)
    if model is None:
        return max_loss, model_text
    print(f"model: {model_text}")
    
    description = model_text + " | " + optimiser_text
    
    # Train/Test & DataLoader
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"train={len(train_dataset)} samples, batch={batch_size} --> {len(train_dataset)/batch_size:.1f} batches/epoch")

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainable = count_trainable_parameters(model)
    print(f"Adam: {trainable:,} trainable parameters") # check this is as expected.

    # Training loop
    start = time.time()
    lastGraph = start
    train_losses = []
    test_losses = []
    
    # Stopping condition
    window     = 5 # check average progress between two windows
    min_change = 0.005 # stop if lossNew/lossOld - 1 < min_change

    if max_overfit >= 1.5:
        window = 10 # allow the model longer to recover from any exploratory excursions.
        
    # Plot a graph of the loss vs epoch at regular intervals
    graph_interval = 5
    
    for epoch in range(0, max_epochs):
        model.train() # ensure we compute gradients
        
#        epoch_start = time.time()
        sum_train_loss = 0
        sum_batches = 0
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
        
            # Forward pass
            loss, _ = model.forward_loss(inputs)
            
            numeric_loss = loss.item() # loss is a tensor
            
            if np.isnan(numeric_loss) or numeric_loss > max_loss:
                print(f"*** Aborting: model exploded! loss={loss:.2f} vs max={max_loss}")
                return max_loss, model_text

            sum_train_loss += numeric_loss * len(inputs)
            sum_batches += len(inputs)

            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
#        elapsed = time.time() - epoch_start
#        print(f"Epoch took {elapsed:.2} sec")
        
        # Store the loss after each epoch:
        approx_train_loss = sum_train_loss / sum_batches # effectively the loss at the previous time step before the most recent back-propagation
        if use_exact_train_loss:
            exact_train_loss = compute_average_loss(model, train_dataset, batch_size) # expensive operation
            pct_error = 100 * (approx_train_loss/exact_train_loss - 1)
            print(f"training loss: exact={exact_train_loss:.2f}, approx={approx_train_loss:.2f}, diff={pct_error:.2f}%")
            approx_train_loss = exact_train_loss
            
        train_losses.append(approx_train_loss)
        test_losses.append(compute_average_loss(model, test_dataset, batch_size)) # this is an acceptable overhead if the test set is several times smaller than the train set.
        if np.isnan(train_losses[-1]) or np.isnan(test_losses[-1]):
            print("Aborting: model returns NaNs :(") # Happens when the learning rate is too high
            return max_loss, model_text

        
        # Progress
        now = time.time()
        total_time = now - start

        # Save the best models (but not too often)
        global last_saved_loss
        if epoch >= 5 and train_losses[-1] < last_saved_loss * 0.95:
            last_saved_loss = train_losses[-1]
            
            # Save the model:
            file_name = "Models/" + model_type # keep over-writing the same file as the loss improves
            print(f"*** Best! loss={last_saved_loss:.2f}, {model_text}, {optimiser_text}")
            print(f"hyper-parameters: {hyper_params}")
            torch.save(model.state_dict(), file_name + ".wab")
            
            # Write the parameters to file:
            with open(file_name+".txt", 'w') as file:
                file.write(str(hyper_params) + "\n\n")
                file.write(model_text + "\n")
                file.write(f"{count_trainable_parameters(model):,} weights & biases\n\n")
                file.write(f"optimiser: {optimiser_text}\n")
                file.write("\n")
                file.write(f"train loss={train_losses[-1]:.1f}, test  loss={test_losses[-1]:.1f}, overfit={train_losses[-1]/test_losses[-1]:.2f}\n")
                file.write(f"time={total_time:.0f} sec, train_size={len(train_dataset)}, batch_size={batch_size}, epoch={epoch} = {total_time/epoch:.1f} sec/epoch\n")
            
            # Generate a test tone:
            resynth, loss = predict_stft(model, sanity_test_stft)
#            norm = (sanity_test_stft[:, :, resynth.size(2)] - resynth).norm()
#            print(f"Resynth loss={loss.item():.2f} for {sanity_test_name}, norm={norm:.2f}")
            save_and_play_audio_from_stft(resynth, sample_rate, stft_hop, f"Results/{model_type} {sanity_test_name} - resynth.wav", False)
            plot_stft("Resynth " + sanity_test_name, resynth, sample_rate, stft_hop)
            

        if verbose and now - lastGraph > graph_interval and len(train_losses) > 1:
            plot_train_test_losses(train_losses, test_losses)
            lastGraph = now
            graph_interval = int(min(3600, 1.5*graph_interval))


        if stop_condition(train_losses, test_losses, window, min_change, max_overfit, total_time):
            break
    
        if epoch < 5: # Test a random sample to show that the code is working from A-Z
            resynth, loss = predict_stft(model, sanity_test_stft)

        if total_time > max_time:
            print("Total time={:.1f} sec exceeds max={:.0f} sec".format(total_time, max_time))
            break
            
        # Early stopping based on average convergence:
        # In practice this didn't work well: the mean can fluctuate to much, even when removing outliers, and it's not sufficiently aggressive.
        # Note: this should really be time-based rather than epoch as models run at different speeds depending on batch size, learning rate and model size.
        # ie: if after 1mn the model is worse than the average at 1mn then give up...
        # In practice, the current implementation doesn't work too well, the stdev can be very high. Could try mean - 0.1 x stdev ? ...
#        if epoch > 30 and epoch % 10 == 0:
#            mean, stdev = compute_epoch_stats(all_test_losses, epoch, 10)
#            loss = test_losses[-1]
#            if mean is not None and loss > mean: # we could make this more aggressive, for example: mean - 0.5 * stdev
#                print(f"Early stopping at epoch={epoch}, test loss={loss:.1f} vs mean={mean:.1f}")
#                break

        # Early stopping: abort if a model is converging too slowly vs the best.
        # Unfortunately this is not in time space, but in epochs.
        # So we could miss out on a model that is slow to train but reaches a better optimal loss.
        # That said, in practice, the models with the lowest loss are those that train quickest at the outset.
        global best_train_losses
        if epoch >= 20 and epoch < len(best_train_losses):
            ratio = train_losses[epoch] / best_train_losses[epoch]
            if ratio > 3:
                print(f"Early stopping at epoch={epoch}, train loss={train_losses[epoch-1]:.1f} vs best={best_train_losses[epoch]:.1f}, ratio={ratio:.1f}")
                return min(best_train_losses) * ratio, description # approximation in order not to mess up the GPR too much.
                break


    # Done!
    
    # Store the best train loss curve, this will be used for early termination when hyper-tuning
    if len(best_train_losses) == 0 or np.min(train_losses) < np.min(best_train_losses):
        best_train_losses = train_losses
    
    # Report the results
    trainL  = train_losses[-1]
    testL   = test_losses[-1]
    elapsed = time.time() - start
    epochs  = len(train_losses)
    print("\n\nFinished Training after {} epochs in {:.1f} sec ({:.2f} sec/epoch), sample duration={:.1f} sec, test loss={:.2f}, train loss={:.2f}, overfit={:.1f}"\
    .format(epochs, elapsed, elapsed/epochs, sample_duration, testL, trainL, testL/trainL))
    
    all_test_losses.append(test_losses)
    all_test_names.append("loss={:.1f}, {}, {}".format(np.min(test_losses), model_text, optimiser_text))
    
    plot_multiple_losses(all_test_losses, all_test_names, 5) # this could become large...
    
    if verbose:
        plot_train_test_losses(train_losses, test_losses)
    
    return np.min(test_losses), description



from IPython.display import HTML, display


def display_custom_link(file_path, display_text=None):

    if display_text is None:
        display_text = file_path

    link_str = f'<a href="{file_path}" target="_blank">{display_text}</a>'
    display(HTML(link_str))    




