import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, gmean, norm
import torch
import time
import inspect
from Debug import *
import math


def is_running_in_jupyter():
    stack = inspect.stack()
    for item in stack:
        if 'IPython' in item[1] or 'ipykernel' in item[1]:
            return True
    return False

is_interactive = is_running_in_jupyter()
print(f"Jupyter={is_interactive}, MatPlotLib.isinteractive()={plt.isinteractive()}")


start_time = time.time()
hour = 3600


def total_time():
    return time.time() - start_time



# Crazy idea: let's make training videos!
import imageio.v2 as imageio
import io
import uuid
import base64

# Generate a random UUID (GUID)
def uuid_to_base64(uuid_value = uuid.uuid4()):
    # Convert UUID to bytes
    uuid_bytes = uuid_value.bytes

    # Encode the bytes to base64
    base64_encoded = base64.urlsafe_b64encode(uuid_bytes)

    # Convert to string and remove '=' padding characters
    return base64_encoded.decode('utf-8').rstrip('=')

unique_id = uuid_to_base64()
print(f"Unique ID: {unique_id}")

class PlotVideoMaker:
    def __init__(self, name, auto_save, pad_time):
        self.images = []
        self.name = name
        self.auto_save = auto_save
        self.last_save = time.time()
        self.needs_saving = False
        self.pad_time = pad_time
        print(f"PlotVideoMaker: {self.name}, auto-save={self.auto_save}")
        
    def add_plot(self, show):
        # Save the current figure as an in-memory image and add to the list
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = imageio.imread(buf)
        self.images.append(image)
        self.needs_saving = True
        buf.close()
        print(f"PlotVideoMaker: {self.name}, frames={len(self.images):,}")
        
        # Optionally display
        if show and is_interactive:
            plt.show()
        else:
            plt.close()
        
        if self.auto_save:
            elapsed = time.time() - self.last_save
            if elapsed > 30:
                self.automatic_save()
                
    def automatic_save(self):
        if self.needs_saving:
            count = len(self.images)
            duration = max(3, count / 10) # use a low FPS
            self.save_video(self.name + " - " + unique_id + ".gif", duration)
            self.last_save = time.time()
            self.needs_saving = False
    
    def save_video(self, file_name, duration):
        # Save the images as an animated GIF
        # Because the GIF cycles, we added some duplicates of the first & last images
        fps = len(self.images) / duration
        
        if False: # for some reason this doesn't work, the GIF doesn't repeat the additional frames!! :(
            duplicate = int(self.pad_time * fps)
            print(f"count={len(self.images)}, fps={fps:.1}, pad_time={self.pad_time}, duplicate={duplicate}")
            save_images = duplicate * [self.images[0]] + self.images + duplicate * [self.images[-1]]
        else:
            save_images = self.images
            
        count = len(save_images)
        duration = count / fps
        file_name = "Videos/" + file_name
        print(f"saving video {file_name}, {count} frames = {duration:.1f} sec @ {fps:.1f} FPS")
        imageio.mimsave(file_name, save_images, duration = duration / count)

#    def __del__(self):
#        self.automatic_save() # crashes.


if __name__ == '__main__':
    plot_video_maker = PlotVideoMaker("SineWaveDemo", False, 1.0)

    # Create some plots independently and add them to the video maker
    x = np.linspace(0, 2 * np.pi, 100)
    frames = 45
    fps = 30
    for i in range(frames):
        #plt.figure()  # Create a new figure for each plot
        y = np.sin(x + 2 * np.pi * i / frames)
        plt.plot(x, y)
        plt.title(f"frame#{i+1:>2}")
        plot_video_maker.add_plot(False)  # Add the current plot to the video maker

    # Save the plots as a GIF
    plot_video_maker.save_video("SineWaveDemo.gif", frames / fps)


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


def plot_train_test_losses(train_losses, test_losses, title):
    assert(len(train_losses) == len(test_losses))
    
    plt.figure(figsize=(10, 5))
        
    plot_loss(train_losses, "Train", "cyan")
    plot_loss(test_losses,  "Test",  "blue")

    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.gca().set_yscale('log')
    plt.title(title + ": loss after {} epochs".format(len(train_losses)))
    plt.legend()
    plt.show()


hyperVideo = PlotVideoMaker("Hyper-Training", True, 0.5)

def plot_multiple_losses(losses, names, min_count, title):
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
        
    title = title + ": loss vs epoch"
    if len(losses) > 1:
        title += f" for {len(losses)} runs ({int(total_time()):,} sec)"
        
    plt.title(title)
    plt.ylabel("Loss (log scale)")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    hyperVideo.add_plot(True)


def plot_hypertrain_loss(loss, names, model_name):
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
        for rank in range(top):
            i = order[rank]
            plt.scatter(i+1, loss[i], marker="o", s=12, c='r')
            plt.text(i+1, loss[i], f"#{rank+1} = {loss[i]:.1f}")
    
    # running average
    window = int(1 + len(loss)/5)
    if window > 1:
        mean = [np.mean(loss[max(0, i - window + 1):i + 1]) for i in range(len(loss))]
        plt.plot(runs, mean, c='cyan', label = f"average of last {window}")
    
    plt.xlabel("Run")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.title(f"{model_name} hyper-parameter optimisation: loss over {len(loss)} runs ({int(total_time()):,} sec)")
    plt.show()
    
    
if __name__ == '__main__':
    import num2words
    N = 100
    plot_hypertrain_loss([np.random.uniform(0, 1) * np.exp(-t/N) for t in range(N)], [num2words(n+1) for n in range(N)], "Test Crash Dummy")


def plot_bar_charts(encodings, names, title):
    assert(len(encodings) == len(names))
    dimensions = len(encodings[0])
    count = len(names)
    x = np.arange(dimensions)

    bar_width = 0.7 / count

    plt.figure(figsize=(12, 6))
    
    e = bar_width * count * 0.05
    var_width = bar_width * count + 2 * e
    
    for j in range(dimensions):
        vx = x[j] - bar_width / 2 - e
        plt.hlines(0, vx, vx + var_width, colors='black')
        
    for i in range(count):
        plt.bar(x + i * bar_width, encodings[i], width=bar_width, label=names[i])

    plt.xticks(x + bar_width * (count - 1) / 2, [f'#{j+1}' for j in range(dimensions)])
    plt.title(title)
    
    if count <= 30:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
    plt.show()


if __name__ == '__main__':
    plot_bar_charts([[1, 2, 3], [2, 4, 8], [-3, 6, 9]], ["counting", "powers", "threes"], "demo")



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

def hide_sub_plot(i):
    axs = plt.gcf().get_axes()
    assert(i < len(axs))
    ax = axs[i]
    ax.axis('off')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_frame_on(False)  # Remove frame around the image


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
        hide_sub_plot(i)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = [torch.rand(57, 150).mul(np.random.uniform(x)) for x in range(11)]
    display_image_grid(images, "Example")
