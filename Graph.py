import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gmean, norm
import torch
from Debug import *

# Heuristic of number of buckets in a histogram
def sturges(N):
    return 1 + math.log2(N) # strictly

def rice(N): # I like this better
    return 2 * (N ** (1/3))


# This code was generated using ChatGPT4 and multiple iterations to fix issues and improve it!
def plot_multiple_histograms_vs_gaussian(series_list, series_names=None):
    plt.figure(figsize=(10, 6))
    
    # Compute global min and max across all series for setting plot limits
    global_min = min([np.min(s) for s in series_list])
    global_max = max([np.max(s) for s in series_list])
    
    # Add some buffer for visibility
    e = 0.05 * (global_max - global_min)
    xmin = global_min - e
    xmax = global_max + e
    
    colors = ['c', 'm', 'y', 'g', 'r', 'b', 'k']
    
    N = max(len(s) for s in series_list)
    bins = int(rice(N))
    ticks = int(5 * rice(N))
    
    for i, series in enumerate(series_list):
        color = colors[i % len(colors)]  # Cycle through colors
        mu, std = np.mean(series), np.std(series)
        
        name = series_names[i] if series_names else i+1  # Use provided name or index
        
        plt.hist(series, bins=bins, density=True, alpha=0.6, color=color, label=f'{name} (mean={mu:.2f}, std={std:.2f})')
        
        x = np.linspace(xmin, xmax, ticks)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, color, linewidth=2)
        
        plt.plot([mu, mu], [0, norm.pdf(mu, mu, std)], color=color, linestyle='--', linewidth=1)
    
    plt.title(", ".join(series_names))
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.show()



def plot_series(arrays, names, bar_chart=False, log_scale=False):

    if len(arrays) != len(names):
        raise ValueError("The number of arrays and names should match.")

    # Set any zero or negative values to 1e-6 if log_scale is True
    if log_scale:
        arrays = [np.maximum(array, 1e-6) for array in arrays]

    if bar_chart:
        # Bar chart plotting
        bar_width = 0.8 / len(arrays)
        for idx, (array, name) in enumerate(zip(arrays, names)):
            positions = np.arange(len(array)) + idx * bar_width
            plt.bar(positions, array, width=bar_width, label=name)
        plt.xticks(np.arange(len(arrays[0])) + 0.4, np.arange(len(arrays[0])))
    else:
        # Line plot
        for array, name in zip(arrays, names):
            plt.plot(array, label=name)

    if log_scale:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)
    plt.show()



def compute_stats_without_outliers(data, min_count):

    if len(data) == 0:
        return None, None

    data = np.array(data)

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    if len(filtered_data) < min_count:
        return None, None
        
    return np.mean(filtered_data), np.std(filtered_data)


# Compute the mean & stdev for a given epoch across multiple runs
def compute_epoch_stats(losses, epoch, min_count):

    epoch_losses = [loss_list[epoch] for loss_list in losses if len(loss_list) > epoch]
    
    return compute_stats_without_outliers(epoch_losses, min_count)


def plot_loss(losses, name=None, colour=None, linewidth = 1):

    epochs = 1 + np.array(range(len(losses)))
    plt.plot(epochs, losses, label=name, color=colour, linewidth=linewidth)
    
    i = np.argmin(losses)
    min_loss = losses[i]
    
    plt.scatter(i+1, min_loss, c=colour, s=8)
    if name is not None:
        plt.text(i+1, min_loss, f"{min_loss:.1f}", color = colour)


def plot_train_test_losses(train_losses, test_losses):
    assert(len(train_losses) == len(test_losses))
    
    plt.figure(figsize=(10, 5))
        
    plot_loss(train_losses, "Train", "cyan")
    plot_loss(test_losses,  "Test",  "blue")

    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.gca().set_yscale('log')
    plt.title("Loss after {} epochs".format(len(train_losses)))
    plt.legend()
    plt.show()


def plot_multiple_losses(losses, names, min_count):
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    
    # Plot all the loss curves
    min_loss = min([min(l) for l in losses])
    
    for loss, name in zip(losses, names):
        isBest = (min(loss) == min_loss)
        if isBest:
            plot_loss(loss, "Best", "cyan", 2)
        else:
            plot_loss(loss)
            
    # Plot mean & stdev
    if len(losses) >= min_count:
        max_epochs = max([len(l) for l in losses])
        step = 5
        epochs = [e for e in range(0, max_epochs, step)]
        stats = [compute_epoch_stats(losses, e, min_count) for e in epochs]
        stats = [s for s in stats if s[0] is not None]
        Ms  = np.array([s[0] for s in stats])
        Xs  = [x+1 for x in range(0, len(Ms)*step, step)]
        assert(len(Xs) == len(Ms))
        plt.plot(Xs, Ms, label = "Mean loss", linewidth=2, c="blue")
        # Plotting the standard-deviations proved too noisy
        if False:
            SDs = np.array([s[1] for s in stats])
            assert(len(Ms) == len(SDs))
            plt.fill_between(Xs, Ms - SDs, Ms + SDs, color='gray', alpha=0.2, label='±1 SD')
        
    title = "Loss vs Epoch"
    if len(losses) > 1:
        title += f" for {len(losses)} runs"
    plt.title(title)
    plt.ylabel("Loss (log scale)")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_hypertrain_loss(loss, names):
    if len(loss) < 2:
        return
    
    assert(len(loss) == len(names))
    loss = np.array(loss)
    
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    runs = [x+1 for x in range(len(loss))]
    
    plt.scatter(runs, loss, marker="o", s=8, c='b', label = "loss")
    order = np.argsort(loss)
    top = 3
    if len(loss) >= top:
        print("\n\nBest Models:")
        for rank in range(top):
            i = order[rank]
            plt.scatter(i+1, loss[i], marker="o", s=12, c='r')
            plt.text(i+1, loss[i], f"#{rank+1} = {loss[i]:.1f}")
            print(f"\t{rank+1}: loss={loss[i]:.1f}, for {names[i]}")
    
    # running average
    window = int(1 + len(loss)/5)
    if window > 1:
        mean = [np.mean(loss[max(0, i - window + 1):i + 1]) for i in range(len(loss))]
        plt.plot(runs, mean, c='cyan', label = f"average of last {window}")
    
    plt.xlabel("Run")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.title(f"Hyper-Parameter Optimisation: loss over {len(loss)} runs")
    plt.show()
    
    
if False:
    from num2words import num2words
    N = 100
    plot_hypertrain_loss([np.random.uniform(0, 1) * np.exp(-t/N) for t in range(N)], [num2words(n+1) for n in range(N)])


def plot_bar_charts(encodings, names, title):
    assert(len(encodings) == len(names))
    vars = len(encodings[0])
    count = len(names)
    x = np.arange(vars)

    bar_width = 0.7 / count

    plt.figure(figsize=(12, 6))
    
    e = bar_width * count * 0.05
    var_width = bar_width * count + 2 * e
    
    for j in range(vars):
        vx = x[j] - bar_width / 2 - e
        plt.hlines(0, vx, vx + var_width, colors='black')
        
    for i in range(count):
        plt.bar(x + i * bar_width, encodings[i], width=bar_width, label=names[i])

    plt.xticks(x + bar_width * (count - 1) / 2, [f'#{j+1}' for j in range(vars)])
    plt.title(title)
    
    if count <= 30:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
    plt.show()


#plot_bar_charts([[1, 2, 3], [2, 4, 8], [-3, 6, 9]], ["counting", "powers", "threes"], "demo")


def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def display_image(ax, image, title, colour_map = 'gray'):
    image = normalize_tensor(image)
    ax.imshow(image, cmap=colour_map)
    ax.axis('off')  # Turn off axis numbers and labels
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_frame_on(False)  # Remove frame around the image
    if title:
        ax.set_title(title)

def display_image_grid(images, title, colour_map = 'gray', min_width=15):
    count = len(images)
    cols = int(np.sqrt(count))
    rows = count // cols
    if rows * cols < count:
        cols += 1

    # Ensure the entire grid is at least `min_width` units wide
    iw = images[0].size(0)
    ih = images[0].size(1)
    
    fig_width = max(min_width, cols)
    fig_height = (fig_width / cols) * rows * iw / ih  # Scale height to maintain square pixels

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(count):
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i]
        display_image(ax, images[i], None, colour_map) #f"Image {i+1}")

    # Hide any unused subplots
    for i in range(count, rows*cols):
        axs.flatten()[i].axis('off')
        axs.flatten()[i].set_xticks([])  # Remove x-axis ticks
        axs.flatten()[i].set_yticks([])  # Remove y-axis ticks
        axs.flatten()[i].set_frame_on(False)  # Remove frame around the image

    plt.show()


#images = [torch.rand(57, 150).mul(np.random.uniform(x)) for x in range(11)]
#display_image_grid(images, "Example")

