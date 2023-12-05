import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from Device import *
import matplotlib.pyplot as plt
from Debug import *


# Interpolates N values exponentially in the range [start, end]
def exponential_interpolation(start, end, N):
    return [start * (end/start) ** (i/(N-1)) for i in range(N)]


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_for_layer(name, layer, input):

    try:
        with torch.no_grad():
            output = layer.forward(input)
            if isinstance(output, tuple):
                output = output[0]

        print(f"{name}: layer={layer}, input={input.shape} --> output={output.shape}")
        return output
            
    except Exception as e:
        print(f"{name}: layer={layer}, input={input.shape} --> error:{e}")
        raise e


def compute_average_loss(model, dataset, batch_size):
    
    model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computations during evaluation
        for batch_idx, (inputs,) in enumerate(data_loader):
            inputs = inputs.to(device)
            loss, _ = model.forward_loss(inputs)
            loss = loss.item()
            
            if np.isnan(loss): # give-up
                raise Exception("model.forward_loss returned NaN :(")
                
            if loss > 1e6: # also give up if the model explodes
                raise Exception(f"model.forward_loss exploded: loss={loss:g} :(")
                
            total_loss += loss * len(inputs)
            total_samples += len(inputs)

    return total_loss / total_samples


def split_dataset(dataset, ratio):
    train_length = int(ratio * len(dataset))
    test_length = len(dataset) - train_length
    return random_split(dataset, [train_length, test_length])


last_progress = time.time()
progress_seconds = 5

def stop_condition(train_losses, test_losses, window, min_change, max_overfit, total, epochs, verbose = False):
    global last_progress, progress_seconds
    
    def delta(losses):
        return 100 * (losses[-1]/losses[-2] - 1) if len(losses) > 1 else 0
        
    def loss_and_change(name, losses):
        old = np.mean(losses[-2*window:-window])
        new = np.mean(losses [-window:-1])
        change = new / old - 1
        
        if verbose:
            print("{} average over {}: old={:.1f}, new={:.1f}, change={:.2f}%".format(name, window, old, new, change))
            
        return new, change

    now = time.time()
    if now - last_progress > progress_seconds:
        last_progress = now
        print("total={:.0f} sec, epoch={} ({:.1f} sec/epoch), train={:.4f} ({:.2f}%), test={:.4f} ({:.2f}%), overfit={:.2f}"\
        .format(total, epochs, total/epochs, train_losses[-1], delta(train_losses), test_losses[-1], delta(test_losses), test_losses[-1]/train_losses[-1]))
        
    if len(test_losses) < 2*window: # Too few epochs
        return False
        
    # Continue if we have an unexpected last-minute improvement, even if the average scores are stuck.
#    if delta(test_losses) < 100*min_change
#        return False
    
    test_loss, test_change = loss_and_change("Test", test_losses)
    train_loss, train_change = loss_and_change("Train", train_losses)
    
    # Here we stop if BOTH scores have stopped changing.
    # This means we can allow overfitting if we want to (for proof of concept).
    # Use max_overfit to stop early once the test loss is stuck.
    if abs(train_change) < min_change and abs(test_change) < min_change:
        return True
    
    overfit = test_loss / train_loss
    
    if epochs > 30 and overfit < 0.5: # this model is garbage
        return True
    
    if verbose:
        print("overfit={:.1f}".format(overfit))
    
    return  test_loss / train_loss > max_overfit


# Compute the mean & stdev for a given epoch across multiple runs
def compute_epoch_stats(losses, epoch, min_count):
    # Extract the losses for the given epoch from each list, if available
    epoch_losses = [loss_list[epoch] for loss_list in losses if len(loss_list) > epoch]

    # Compute average and standard deviation if there are any losses for this epoch
    if len(epoch_losses) >= min_count:
        avg_loss = np.mean(epoch_losses)
        stdev_loss = np.std(epoch_losses)
        return avg_loss, stdev_loss
    else:
        return None, None


def plot_multiple_losses(losses, names, min_count):
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    
    # Plot all the loss curves
    min_loss = min([min(l) for l in losses])
    
    for loss, name in zip(losses, names):
        isBest = (min(loss) == min_loss)
        Xs = [x+1 for x in range(len(loss))]
        if isBest:
            plt.plot(Xs, loss, label = "best: " + name, linewidth=2, alpha=1, c="cyan")
        else:
            plt.plot(Xs, loss, linewidth=1, alpha=0.5)
            
    # Plot mean & stdev
    max_epochs = max([len(l) for l in losses])
    step = 10
    epochs = [e for e in range(0, max_epochs, step)]
    stats = [compute_epoch_stats(losses, e, min_count) for e in epochs]
    stats = [s for s in stats if s[0] is not None]
    Ms  = np.array([s[0] for s in stats])
    SDs = np.array([s[1] for s in stats])
    assert(len(Ms) == len(SDs))
    Xs  = [x+1 for x in range(0, len(Ms)*step, step)]
    assert(len(Xs) == len(Ms))
    plt.plot(Xs, Ms, label = "Mean loss", linewidth=2, c="blue")
    plt.fill_between(Xs, Ms - SDs, Ms + SDs, color='gray', alpha=0.2, label='±1 SD')
        
    plt.title(f"Test Loss vs Epoch for {len(losses)} runs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def random_exponential_decay_list(N):
    start_value = np.random.uniform(0.5, 2.0)
    decay_rate = np.random.uniform(0.01, 0.05)
    return start_value * np.exp(-decay_rate * np.arange(N)) #* (1 + noise * np.random.uniform(0, 1, N)))

def test_loss_chart():
    lengths = range(10, 200, 5)
    examples = [random_exponential_decay_list(n) for n in lengths]
    names = ["example#"+str(n) for n in lengths]
    plot_multiple_losses(examples, names, 5)

#test_loss_chart()


# Computes the number of parameters of stacked fully-connected layers
def fully_connected_size(layer_sizes):

    total_params = 0
    
    for i in range(len(layer_sizes) - 1):
        total_params += (layer_sizes[i] * layer_sizes[i+1]) + layer_sizes[i+1]
        
    return total_params


# Builds multiple fully-connected layers with ReLU() in between:
def build_sequential_model(layer_sizes):
    layers = []
    
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:  # Add ReLU activation for all but the last layer
            layers.append(nn.ReLU())
            
    return nn.Sequential(*layers)
    
# Interpolates a list of layer sizes form start input to and end output, with a given depth layers and a power ratio
# If ratio = 1, the interpolation is linear.
# If ratio < 1, the intermediate layers will tend towards the end size.
# If ratio > 1, the intermediate layers will tend towards the start size.
# This allows us to parameterise the construction of N-layer MLPs, whilst biasing the layer sizes to the start dimension or the end
# dimensions. Assuming start & end are fixed, we only have 2 parameters to tune for the shape of the MLP: depth & ratio.

def interpolate_layer_sizes(start, end, depth, ratio):
    assert(depth > 0)
    
    if depth == 1:
        return [start, end]

    layers=[]
    for i in range(depth):
        t = i / (depth - 1)
        t = t ** ratio
        layers.append(int(start + t * (end - start)))
    
    #print(f"start={start}, end={end}, depth={depth}, ratio={ratio:.2f} --> layers={layers}")
    return layers
