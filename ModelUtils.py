from Device import *
from Debug import *
from Graph import *

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np


default_activation_function = nn.ReLU()
#default_activation_function = nn.GELU()


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
            
    except BaseException as e:
        print(f"{name}: layer={layer}, input={input.shape} --> error:{e}")
        raise e


def compute_average_loss(model, dataset, batch_size):
    
    model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computations during evaluation
        for batch_idx, inputs in enumerate(data_loader):
            inputs = inputs.to(device)
            loss, _ = model.forward_loss(inputs)
            loss = loss.item()
            
            total_loss += loss * len(inputs)
            total_samples += len(inputs)

    return total_loss / total_samples


def split_dataset(dataset, ratio):
    train_length = int(ratio * len(dataset))
    test_length = len(dataset) - train_length
    return random_split(dataset, [train_length, test_length])


last_progress = time.time()
progress_seconds = 5

def stop_condition(train_losses, test_losses, window, min_change, max_overfit, total, verbose = False):
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

    epochs = len(train_losses)
    assert(epochs == len(test_losses))
    
    now = time.time()
    if now - last_progress > progress_seconds:
        last_progress = now
        print("total={:.1f} sec, epoch={} ({:.1f} sec/epoch), train={:.2f} ({:.2f}%), test={:.2f} ({:.2f}%), overfit={:.2f}"\
        .format(total, epochs, total/epochs, train_losses[-1], delta(train_losses), test_losses[-1], delta(test_losses), test_losses[-1]/train_losses[-1]))
        
    if len(test_losses) < 2*window: # Too few epochs
        return False
        
    # Continue if we have an unexpected last-minute improvement, even if the average scores are stuck.
#    if delta(test_losses) < 100*min_change
#        return False
    
    test_loss,  test_change  = loss_and_change("Test", test_losses)
    train_loss, train_change = loss_and_change("Train", train_losses)
    
    # Here we stop if BOTH scores have stopped changing.
    # This means we can allow overfitting which is helpful in some use-cases.
    # Use max_overfit to stop early once the test loss is stuck.
    if abs(train_change) < min_change and abs(test_change) < min_change:
        print("Training stalled.")
        return True
    
    overfit = test_loss / train_loss
    if verbose:
        print("overfit={:.2f}".format(overfit))
    
    if epochs > 30 and overfit < 0.5: # this model is garbage
        print(f"Model doesn't generalise: overfit={overfit:.2f}")
        return True
    
    if test_loss / train_loss > max_overfit:
        print(f"Model is overfitting: overfit={overfit:.2f} vs max={max_overfit:.2f}")
        return True
        
    # Keep going...
    return False






def random_exponential_decay_list(N):
    start_value = np.random.uniform(0.5, 2.0)
    decay_rate = np.random.uniform(0.01, 0.05)
    return start_value * np.exp(-decay_rate * np.arange(N)) #* (1 + noise * np.random.uniform(0, 1, N)))


def test_loss_chart():
    lengths = range(10, 200, 5)
    examples = [random_exponential_decay_list(n) for n in lengths]
    names = ["example#"+str(n) for n in lengths]
    plot_multiple_losses(examples, names, 5, "Text Example")

#test_loss_chart()


# Computes the number of parameters of stacked fully-connected layers
def fully_connected_size(layer_sizes):

    total_params = 0
    
    for i in range(len(layer_sizes) - 1):
        total_params += (layer_sizes[i] * layer_sizes[i+1]) + layer_sizes[i+1]
        
    return total_params


# Builds multiple fully-connected layers with and activation function in between:
def sequential_fully_connected(layer_sizes, final_activation):

    if len(layer_sizes) < 2:
        return []

    layers = []
    
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:  # Add activation for all but the last layer
            layers.append(default_activation_function)
    
    assert(len(layers) == 2*(len(layer_sizes)-1) - 1)
    
    if final_activation is not None:
        layers.append(final_activation)
    
    return nn.Sequential(*layers)
    
# Interpolates a list of layer sizes form start input to and end output, with a given depth layers and a power ratio
# If ratio = 1, the interpolation is linear.
# If ratio < 1, the intermediate layers will tend towards the end size.
# If ratio > 1, the intermediate layers will tend towards the start size.
# This allows us to parameterise the construction of N-layer MLPs, whilst biasing the layer sizes to the start dimension or the end
# dimensions. Assuming start & end are fixed, we only have 2 parameters to tune for the shape of the MLP: depth & ratio.

def interpolate_layer_sizes(start, end, depth, ratio):
    assert(depth > 0)
    
    layers = [start]
    for i in range(depth):
        t = (i+1) / depth
        t = t ** ratio
        layers.append(int(start + t * (end - start)))
    
    #print(f"start={start}, end={end}, depth={depth}, ratio={ratio:.2f} --> layers={layers}")
    assert(len(layers) == depth + 1)
    assert(layers[0] == start)
    assert(layers[-1] == end)
    
    return layers


# Number of trainaible parameters in an RNN
def rnn_size(input_size, hidden_size, num_layers):
    first_layer_params = (input_size * hidden_size) + (hidden_size ** 2) + (2 * hidden_size)

    additional_layer_params = (hidden_size ** 2) + (hidden_size ** 2) + (2 * hidden_size)

    total_params = first_layer_params + ((num_layers - 1) * additional_layer_params)
    
    return total_params


def conv1d_size(input_channels, num_kernels, kernel_size):
    return (kernel_size * input_channels + 1) * num_kernels


def load_weights_and_biases(model, file_name):
        print(f"{model.__class__.__name__}: loading weights & biases from file '{file_name}'")
        model.load_state_dict(torch.load(file_name))
    
    
def freeze_model(model):
    print(f"Freezing model {model.__class__.__name__}")
    for name, param in model.named_parameters():
        #print(f"\tfreezing: {name}")
        param.requires_grad = False


count = 0
last_count = 0
do_display_hiddens = False

def set_display_hiddens(onOff):
    global do_display_hiddens
    do_display_hiddens = onOff
    
def periodically_display_2D_output(hiddens):

    global count, last_count, do_display_hiddens, is_interactive
    
    if do_display_hiddens and is_interactive:
        count += hiddens.size(0)
        
        if count - last_count > 10_000: # approx every 10 epochs
            last_count = count
            hiddens = hiddens.detach().cpu()
            width = hiddens[0].size(0)
            height = hiddens[0].size(1)
            display_image_grid(hiddens.transpose(2, 1), f"Hidden outputs {width} x {height}", "magma")



def compute_final_learning_rate(name, losses, window):
    count = len(losses)
    
    if count < window:
        return 1.0 # ideally the final_learning_rate should be < 0.
        
    ratios = [ (losses[i] / losses[i - 1]) - 1 for i in range(count - window + 1, count)]
    average = np.mean(ratios)
    print(f"{name}: final learning-rate={average*100:.2f}%")
    return average


if __name__ == '__main__':
    window = 6
    
    for i in range(0, 20, 3):
        data = [20-i for i in range(i)]
        compute_final_learning_rate(f"Example#{i+1}", data, window)


def model_output_shape_and_size(model, input_shape):
    model.float()
    model.to(device)

    input = torch.randn(input_shape).to(device)
    output = model(input.unsqueeze(0)).squeeze(0)
    size = output.numel()
    shape = tuple(output.shape)
    #print(f"Model output: shape={shape}, size: {size:,}")
    return shape, size

