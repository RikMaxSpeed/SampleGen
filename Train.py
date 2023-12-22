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

def is_incremental_vae(model_name):
    return "VAE_Incremental" in model_name


def generate_training_stfts(how_many):
    global sanity_test_stft, sanity_test_name, train_dataset, test_dataset
                
     # Augmentation is used if this exceeds the number of real available samples
    stfts, file_names = get_training_stfts(None)
    
    count = len(stfts)
    if how_many is None:
        how_many = count
    
    if False:
        lengths = np.array([x.shape[1] for x in stfts])
        plot_multiple_histograms_vs_gaussian([lengths * stft_hop / sample_rate], ["Sample Durations (seconds)"])

    # Pick an example file to sanity check that everything is behaving from A-Z
    for i in range(len(file_names)):
        if "grand piano c3" in file_names[i].lower():
            sanity_test_stft = stfts[i]
            sanity_test_name = file_names[i]
            break


    stfts = convert_stfts_to_inputs(stfts)
    count = stfts.size(0)
    print(f"{count} STFTs")
    #display_average_stft(stfts, True)

    # Find key samples to encode
    if how_many <= count/3:
        stfts = select_diverse_tensors(stfts, file_names, how_many).to(device)

    if stfts.size(0) > how_many: # truncate if too many
        stfts = stfts[:how_many, : , :]

    # Convert into train & test datasets
    ratio = 0.8
    train_stfts, test_stfts = split_dataset(stfts, ratio)
    
    # Training set is kept completely separate from Test when augmenting.
    train_size = int(how_many * ratio)
    if how_many is not None and len(train_stfts) < how_many * ratio:
        train_stfts = augment_stfts(train_stfts, int(how_many * ratio))

    train_dataset = train_stfts
    test_dataset  = test_stfts
    print(f"Using train={len(train_dataset)} samples, test={len(test_dataset)} samples.")

    return len(train_dataset), len(test_dataset)
    
# If training an incremental VAE, we encode the STFTs just once using the auto-encoder
def encode_stfts(model, name, stfts):
    if len(stfts[0].shape) == 1:
        return stfts # already encoded

    print(f"Encoding {name} {len(stfts)} STFTs")
    return [model.encode(stft.unsqueeze(0)).squeeze(0) for stft in stfts] # add/remove batch dimension


# Hyper-parameter optimisation
last_saved_loss = 200 # don't bother saving models above this threshold

# Keep track of all the test-losses over multiple runs, so we can learn how to terminate early on poor hyper-parameters.
all_test_losses = []
all_test_names = []

# Compare the training loss to the best we've found, and abort if it's too far off.
best_train_losses = []

use_exact_train_loss = False # Setting to True is more accurate but very expensive in CPU time


def reset_train_losses():
    global all_test_losses, all_test_names, best_train_losses, last_saved_loss
    all_test_names = []
    all_test_losses = []
    best_train_losses = []
    last_saved_loss = 200


# Main entry point for training the model
def train_model(model_type, hyper_params, max_epochs, max_time, max_params, max_overfit, max_loss, verbose, load_existing):
    global train_dataset, test_dataset

    # We split the hyper-params into optimiser parameters & model parameters:
    opt_params   = hyper_params[:2]
    model_params = hyper_params[2:]
    
    # Optmiser parameters:
    batch, learning_rate = opt_params
    batch_size = int(2 ** batch) # convert int64 to int32
    
    if learning_rate < 0: # new: convert to 10^lr, but old LRs are still supported
        learning_rate = 10.0 ** learning_rate
    
    learning_rate *= batch_size # see https://www.baeldung.com/cs/learning-rate-batch-size
    weight_decay = 0
    optimiser_text = f"Adam batch={batch_size}, learning_rate={learning_rate:.2g}, weight_decay={weight_decay:.2g}"
    print(f"optimiser: {optimiser_text}")
    
    # Create the model
    if load_existing:
        model, model_text, model_params, model_size = load_saved_model(model_type)
    else:
        model, model_text, model_size = make_model(model_type, model_params, max_params, verbose)
    
    if model is None:
        return model_text, model_size, max_loss, 1.0
        
    trainable = count_trainable_parameters(model)
    model_text += f" (params={model_size:,}, compression={model.compression:.1f}x)"
    description = model_text + " | " + optimiser_text
    print(f"model: {model_text}")

    # Optimisation for Incremental VAE: encode the STFTs using the auto_encoder layers only.
    is_vae = is_incremental_vae(model_type)

    if is_vae:
        # We will only be training the inner VAE, so we first encode the STFTs
        active_model = model.vae
        train_dataset = encode_stfts(model.auto_encoder, "Train", train_dataset)
        test_dataset = encode_stfts(model.auto_encoder, "Test", test_dataset)
    else:
        active_model = model

    # Train/Test & DataLoader
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"train={len(train_dataset)} samples, batch={batch_size} --> {len(train_dataset)/batch_size:.1f} batches/epoch")

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Adam: {trainable:,} trainable parameters") # check this is as expected.
    
    # Training loop
    start = time.time()
    lastGraph = start
    train_losses = []
    test_losses = []
    
    # Stopping condition
    window     = 5 # check average progress between two windows
    min_change = 0.005 # stop if lossNew/lossOld - 1 < min_change

    if max_overfit >= 1.9:
        window = 15 # allow the model longer to recover from any exploratory excursions.
        
    # Plot a graph of the loss vs epoch at regular intervals
    graph_interval = 5
    
    for epoch in range(0, max_epochs):
        active_model.train() # ensure we compute gradients
        
        sum_train_loss = 0
        sum_batches = 0
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
        
            # Forward pass
            loss, _ = active_model.forward_loss(inputs)
            
            numeric_loss = loss.item() # loss is a tensor
            
            if np.isnan(numeric_loss) or numeric_loss > max_loss:
                print(f"*** Aborting: model exploded! loss={loss:.2f} vs max={max_loss}")
                loss = np.min(train_losses) if len(train_losses) else max_loss
                return description, model_size, loss, 9.99

            sum_train_loss += numeric_loss * len(inputs)
            sum_batches += len(inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Store the loss after each epoch:
        train_loss = sum_train_loss / sum_batches # effectively the loss at the previous time step before the most recent back-propagation
        if use_exact_train_loss:
            exact_train_loss = compute_average_loss(active_model, train_dataset, batch_size) # expensive operation
            pct_error = 100 * (train_loss/exact_train_loss - 1)
            print(f"training loss: exact={exact_train_loss:.2f}, approx={train_loss:.2f}, diff={pct_error:.2f}%")
            train_loss = exact_train_loss
            
        test_loss = compute_average_loss(active_model, test_dataset, batch_size) # this is an acceptable overhead if the test set is several times smaller than the train set.
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if np.isnan(train_losses[-1]) or np.isnan(test_losses[-1]):
            print("Aborting: model returns NaNs :(") # High learning rate or unstable model?
            return description, model_size, np.min(train_losses), 9.99

        
        # Progress
        now = time.time()
        total_time = now - start

        # Save the best models (but not too often)
        global last_saved_loss
        if train_losses[-1] < last_saved_loss * 0.95:
            last_saved_loss = train_losses[-1]
            
            # Save the model:
            file_name = "Models/" + model_type # keep over-writing the same file as the loss improves
            print(f"\n*** Best! loss={last_saved_loss:.2f}")
            print(f"{model_text}\n{optimiser_text}\nhyper-parameters: {hyper_params}")
            torch.save(model.state_dict(), file_name + ".wab")

            say_out_loud(f"Best is {last_saved_loss:.2f}".replace(".", " spot "))

            # Write the parameters to file:
            with open(file_name+".txt", 'w') as file:
                file.write(str(hyper_params) + "\n\n")
                file.write(model_text + "\n")
                file.write(f"{count_trainable_parameters(model):,} weights & biases\n\n")
                file.write(f"optimiser: {optimiser_text}\n")
                file.write("\n")
                file.write(f"train loss={train_loss:.2f}, test loss={test_loss:.2f}, overfit={test_loss/train_loss:.2f}\n")
                file.write(f"time={total_time:.0f} sec, train_size={len(train_dataset)}, batch_size={batch_size}, epoch={epoch} = {total_time/(epoch+1):.1f} sec/epoch\n")
                file.write(f"\n{active_model}\n")

            # Generate a test tone:
            resynth, loss = predict_stft(model, sanity_test_stft)
            print(f"Resynth {sanity_test_name}: loss={loss:.2f}")
            save_and_play_audio_from_stft(resynth, sample_rate, stft_hop, f"Results/{model_type} {sanity_test_name} - resynth.wav", False)
            
            # This now saves to video too
            plot_stft(f"{sanity_test_name}, loss={loss:.2f} @ epoch {epoch}", resynth, sample_rate, stft_hop)
            print("\n")

        if verbose and now - lastGraph > graph_interval and len(train_losses) > 1:
            if is_interactive:
                plot_train_test_losses(train_losses, test_losses, model_type)
            lastGraph = now
            graph_interval = int(min(hour, 1.5 * graph_interval)) # less & less frequently!


        if stop_condition(train_losses, test_losses, window, min_change, max_overfit, total_time):
            break
    
        if epoch < 5: # Test a random sample to show that the code is working from A-Z
            resynth, loss = predict_stft(model, sanity_test_stft)

        if total_time > max_time:
            print("Total time={:.1f} sec exceeds max={:.0f} sec".format(total_time, max_time))
            break


        # Early stopping: abort if a model is converging too slowly vs the best.
        # Unfortunately this is not in time space, but in epochs.
        # So we could miss out on a model that is slow to train but reaches a better optimal loss.
        # That said, in practice, the models with the lowest loss are those that train quickest at the outset.
        global best_train_losses
        if epoch >= 20 and epoch < len(best_train_losses):
            ratio = train_losses[epoch] / best_train_losses[epoch]
            if ratio > 3:
                print(f"Early stopping at epoch={epoch}, train loss={train_losses[epoch-1]:.2f} vs best={best_train_losses[epoch]:.2f}, ratio={ratio:.1f}")
                return description, model_size, min(best_train_losses) * ratio, compute_final_learning_rate("Train", train_losses, window) # approximation in order not to mess up the GPR too much.

    # Done!
    if epoch == max_epochs-1:
        print(f"Reached max epochs={max_epochs}")

    # Store the best train loss curve, this will be used for early termination when hyper-tuning
    if len(best_train_losses) == 0 or np.min(train_losses) < np.min(best_train_losses):
        best_train_losses = train_losses
    
    # Report the results
    trainL  = train_losses[-1]
    testL   = test_losses[-1]
    elapsed = time.time() - start
    epochs  = len(train_losses)
    
    print("\n\nFinished Training after {} epochs in {:.1f} sec ({:.2f} sec/epoch), sample duration={:.1f} sec, test loss={:.2f}, train loss={:.2f}, overfit={:.2f}"\
    .format(epochs, elapsed, elapsed/epochs, sample_duration, testL, trainL, testL/trainL))

    if elapsed > 300: # don't blab about failed attempts
        say_out_loud(f"Training stopped at epoch {len(train_losses)}, after {elapsed:.1f} seconds, loss {np.min(train_losses):.2f}")

    train_rate = compute_final_learning_rate("Train", train_losses, window)
    test_rate = compute_final_learning_rate("Test", test_losses, window)

    all_test_losses.append(test_losses)
    all_test_names.append("loss={:.2f}, {}, {}".format(np.min(test_losses), model_text, optimiser_text))
    
    plot_multiple_losses(all_test_losses, all_test_names, 5, model_type) # can have 100+ curves.
    
    if verbose and is_interactive:
        plot_train_test_losses(train_losses, test_losses, model_type)



    # We return the Test Loss: ultimatley we're looking for the model that trains best on the training set. Maximum overfit is handled in the stopping condition.
    return description, model_size, np.min(train_losses), train_rate
