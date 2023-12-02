import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.tri as tri
from scipy.stats import qmc, gmean, norm
#from scipy.spatial import ConvexHull


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
        
        plt.hist(series, bins=bins, density=True, alpha=0.6, color=color, label=f'{name} (mean={mu:.4f}, std={std:.4f})')
        
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



def plot_losses(train_losses, test_losses):

    def plot_loss(name, losses, colour, offset):
        plt.plot(epochs, losses, label = name, color=colour)
        
        i = np.argmin(losses)
        l = losses[i]
        
        plt.scatter(i+1, l, label=None, c=colour, s=8)
        plt.text(i+1, l*(1+offset), "{:.2f}".format(l), color = colour)


    count = len(train_losses)
    assert(count == len(test_losses))
    
    plt.figure(figsize=(10, 5))
    epochs = 1 + np.array(range(count))
    
    plot_loss("Train", train_losses, "cyan", -0.00)
    plot_loss("Test",  test_losses,  "blue", +0.00)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.gca().set_yscale('log')
    plt.title("Loss after {} epochs".format(count))
    plt.legend()
    plt.show()



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

