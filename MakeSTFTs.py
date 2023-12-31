import os
import numpy as np
import librosa
import torch
import pickle
import time
import sys
from AudioUtils import *
from Device import *
from Debug import *


mu_law = MuLawCodec(8) # Yet another hyper-parameter but we can't tune this one as it's outside the model's loss function.
#mu_law = None

amps = AmplitudeCodec()
#amps = None

# Configure the Audio -> STFT conversion
sample_rate = 44100
nyquist = sample_rate // 2
stft_size = 1024 # tried 512
stft_buckets = 2 * stft_size # full frequency range
overlap = 0.5 # 50% gives best quality, 75% has notable artefacts, 66% is low-quality
stft_hop = int(stft_buckets * overlap)

if amps is None:
    max_freq = nyquist // 2 # strip the high frequencies
    freq_buckets = 2 * int(max_freq * 2 * stft_size / sample_rate)
else:
    max_freq = nyquist
    freq_buckets = stft_size + 1


sample_duration = 1.95 # seconds (many of the samples in the library are approx 2 seconds and we get an off-by-one error on the length
sequence_length = int(sample_duration * sample_rate / stft_hop)

print(f"Using sample rate={sample_rate} Hz, FFT={stft_buckets} buckets, hop={stft_hop} samples, duration={sample_duration:.1f} sec = {sequence_length:,} time steps")
print(f"Max frequency={max_freq} Hz --> freq_buckets={freq_buckets}")

def plot_amplitudes_vs_frequency(amplitudes, name):
    global sample_rate, stft_size, stft_buckets
    frequencies = [i * sample_rate / stft_buckets for i in range(len(amplitudes))]
        
    i = np.argmax(amplitudes)
    f = frequencies[i]
    a = amplitudes[i]
    plt.figure(figsize=(9, 5))
    plt.scatter(f, a)
    plt.text(f, a, f"max={f:.1f}Hz")
    plt.plot(frequencies, amplitudes)
    plt.title(f"Amplitude vs Frequency for {name}")
    plt.xscale("log")
    fs = [note_frequency(midi) for midi in range(24, 127, 12)]
    plt.xlim(fs[0], fs[-1])
    plt.xticks(fs, [f"{f:.0f} Hz" for f in fs])
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.show()


def out_of_range(freq, mult):
    return abs(freq / (mult * middleCHz) - 1) > 0.1
  
  
def exclude_frequency(freq):
    return out_of_range(freq, 0.5) and out_of_range(freq, 1) and out_of_range(freq, 2)


def lowest_frequency(stft, sample_rate, name, verbose):
    assert(stft.shape[0] == stft_size + 1)
    amplitudes = np.abs(stft).sum(axis=1)
        
    amplitudes[0] = 0 # Ignore the constant offset
    
    i = np.argmax(amplitudes)
    f = i * sample_rate / stft_buckets
    if not exclude_frequency(f):
        return f
    
    maxAmp = np.max(amplitudes[0 : len(amplitudes) // 4])
    
    for i in range(1, len(amplitudes) - 1):
        f = i * sample_rate / stft_buckets
        if 30 < f < 4*middleCHz:
            a = amplitudes[i]
            if a > maxAmp / 8:
                if amplitudes[i-1] < a > amplitudes[i+1]: # peak
                
                    if verbose and exclude_frequency(f):
                        plot_amplitudes_vs_frequency(amplitudes, name)
            
                    return f
    
    return 0


def compute_stft_from_file(file_name, verbose):
    sr, stft = compute_stft_for_file(file_name, stft_buckets, stft_hop)
    assert(stft.shape[0] == stft_size + 1)
    
    low_hz = lowest_frequency(stft, sr, file_name, verbose)

    return sr, torch.tensor(stft), low_hz


def display_time(start, count, unit):
        elapsed = time.time() - start

        if elapsed > count:
            print("processed {} {}s in {:.1f} sec = {:.1f} sec/{}".format(count, unit, elapsed, elapsed/count, unit))
        else:
            print("processed {} {}s in {:.1f} sec = {:.1f} {}/sec".format(count, unit, elapsed, count/elapsed, unit))


    
    
def gather_stfts_from_directory(directory, notes, requiredSR, verbose):
    """Loops over all .wav files in the given directory and computes their STFTs"""

    start = time.time()
    stft_tensors = []
    file_names = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "Results" in root:
                continue
                
            if file.endswith('.wav'):
                for note in notes:
                    if note in file:
                        filepath = os.path.join(root, file)
                        
                        try:
                            #print("\n\nReading: ", filepath)
                            sr, stft_tensor, low_hz = compute_stft_from_file(filepath, verbose)
                            
                            if sr != requiredSR:
                                print("Skipping {}: sample rate={} Hz".format(filepath, sr))
                            elif exclude_frequency(low_hz):
                                print("Skipping {}: lowest frequency={:.1f} Hz".format(filepath, low_hz))
                            else:
                                stft_tensors.append(stft_tensor)
                                file_names.append(file)
                                print("#{}: {}".format(len(stft_tensors)+1, file))
                                        
                        except AssertionError as e:
                            print("Assertion failed in file {}: {}".format(filepath, e))
                            sys.exit()
                            
                        except BaseException as e:
                            print("Error reading file {}: {}".format(filepath, e))
    
    print("\nDone!!")
    display_time(start, len(stft_tensors), "file")

    return stft_tensors, file_names


def save_to_file(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def load_from_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


stft_file = f"STFT {sample_rate} Hz, size={stft_size}, hop={stft_hop}.pkl"
    
def make_STFTs(verbose):
    #notes = ["C5", "C4", "C3", "C2"]
    notes = ["C3", "C4"] # There's confusion over what C4 means, so we have a frequency check instead
    stft_list, file_names = gather_stfts_from_directory("../WaveFiles", notes, 44100, verbose)
    save_to_file((stft_list, file_names), stft_file)



if os.path.exists(stft_file):
    print("STFT file already created: {stft_file}")
else:
    print("STFT not found: {stft_file}")
    make_STFTs(False)

    
def load_STFTs():
    stfts, file_names = load_from_file(stft_file)
    print("Loaded {} STFTs from {}".format(len(stfts), stft_file))
    return stfts, file_names


def adjust_stft_length(stft_tensor, target_length):
    
    assert(stft_size <= stft_tensor.shape[0] <= stft_size + 1)
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


def convert_to_reals(complex):
    output = torch.zeros(2 * complex.size(0), complex.size(1), dtype=torch.float32)
    output[0::2] = complex.real
    output[1::2] = complex.imag
    return output

def convert_to_complex(reals):
    #return torch.complex(tensor[0::2], tensor[1::2]) # Not implemented on MPS :(
    
    # Alternative implementation: this was a pig to get right!
    real_parts = reals[0::2]  # Even rows: real parts
    imag_parts = reals[1::2]  # Odd rows: imaginary parts

    N, M = real_parts.shape
    combined_tensor = torch.zeros(N, M, 2, dtype=torch.float32)
    combined_tensor[:, :, 0] = real_parts  # Real parts
    combined_tensor[:, :, 1] = imag_parts  # Imaginary parts

    output_tensor = torch.view_as_complex(combined_tensor)
    return output_tensor


def display_min_max(name, stft):
    min = stft.min()
    max = stft.max()
    print(f"{name}: min={min:.3f}, max={max:.3f}")


def remove_low_magnitudes(stft, db):
    assert db < 0
    threshold = dB_to_amplitude(db)
    stft[stft < threshold] = 0
    return stft

def convert_stft_to_input(stft):
    assert(stft.dtype == torch.complex64)
    assert(len(stft.shape) == 2)
    assert(stft.shape[0] == stft_size + 1)
    
    stft = adjust_stft_length(stft, sequence_length)
    
    # Normalise to [-1, 1]
    stft /= torch.max(stft.abs())

    if amps is None:
        # Truncate the frequency range
        stft = stft[:freq_buckets//2,:] # complex, so divide by 2.
        stft = convert_to_reals(stft)
    else:
        stft = amps.encode(stft)
        stft = remove_low_magnitudes(stft, -70)
        
    assert(stft.size(0) == freq_buckets)
    assert(stft.size(1) == sequence_length)
    
    # Mu-Law
    if mu_law is not None:
        stft = mu_law.encode(stft)

    assert(stft.dtype == torch.float32)
    return stft.to(device)


def convert_stfts_to_inputs(stfts):
    return torch.stack([convert_stft_to_input(stft) for stft in stfts]).to(device)


def convert_stft_to_output(stft):
    assert(stft.dtype == torch.float32)

    if amps is None:
        stft = torch.clamp(stft, min=0, max=1)

    if mu_law is not None:
        stft = mu_law.decode(stft)
    
    if amps is None:
        # Re-append truncated frequencies
        assert(stft.size(0) == freq_buckets)
        missing_buckets = torch.full((stft_buckets - freq_buckets +2, sequence_length), 0.0, device=device)
        stft = torch.cat((stft, missing_buckets), dim = 0)
        assert(stft.size(0) == stft_buckets + 2)
        
        stft = convert_to_complex(stft)
        global maxAmp
        stft = stft.cpu().detach() * maxAmp

    else:
        stft = stft.cpu().detach()
        stft = stft.clamp(min=0, max=1)
        stft = stft * maxAmp # re-amplify
        stft = remove_low_magnitudes(stft, -50) # more aggressive
        iterations = 50
        stft = torch.tensor(recover_audio_from_magnitude(stft, stft_buckets, stft_hop, sample_rate, iterations))
        assert(stft.size(0) == stft_size + 1)
        assert(stft.size(1) == sequence_length)
    
    assert(stft.dtype == torch.complex64)
    return stft.numpy()



def test_stft_conversions(file_name):
    sr, stft = compute_stft_for_file(file_name, stft_buckets, stft_hop)
    debug("stft.raw", stft)
    assert(stft.shape[0] == stft_size + 1)
    
    stft = stft[:, :sequence_length]
    debug("truncated", stft)
    
    if False: # Generate a synthetic spectrum
        for f in range(stft.shape[0]):
            for i in range(stft.shape[1]):
                stft[f, i] = 75*np.random.uniform() if np.random.uniform() > 0.9 else 0
    
    amp = np.max(np.abs(stft))
    print(f"original max={amp:.2f}")
    
    plot_stft(file_name, stft, sr, stft_hop)
    
    tensor = torch.tensor(stft)
    debug("original", tensor)
    input = convert_stft_to_input(tensor)
    debug("input", input)
    
    output = convert_stft_to_output(input)
    debug("output", output)
    
    
    global maxAmp
    output *= amp / maxAmp
    
    plot_stft("Resynth " + file_name, output, sr, stft_hop)
    save_and_play_audio_from_stft(output, sr, stft_hop, "Results/resynth-" + os.path.basename(file_name), True)

    diff = np.abs(output - stft)
    debug("diff", diff)
    plot_stft("Diff", diff, sr, stft_hop)
    if False: # Print the differences
        for f in range(stft.shape[0]):
            for t in range(stft.shape[1]):
                d = diff[f, t]
                if d > 0.1:
                    print("f={}, t={}, diff={:.3f}, stft={:.3f}, output={:.3f}".format(f, t, d, stft[f, t], output[f, t]))


if __name__ == '__main__':
    #test_stft_conversions("Samples/Piano C4 Major 13.wav")
    test_stft_conversions("/Users/Richard/Coding/WaveFiles/Essential Synths/Multis/Hoover/Hoover C4.wav")


def display_average_stft(stfts, playAudio):
    mean = stfts.mean(dim=0)
    output = convert_stft_to_output(mean)
    plot_stft("Average STFT", output, sample_rate, stft_hop)
    save_and_play_audio_from_stft(output, sample_rate, stft_hop, "Results/MeanSTFT.wav", playAudio)



def select_diverse_tensors(tensor_array, names, N):
    """Select 'N' most diverse tensors from a tensor of tensors using PyTorch."""
    if N <= 0 or tensor_array.nelement() == 0:
        return torch.tensor([])

    # Compute the average tensor of the entire dataset
    average_tensor = torch.mean(tensor_array, dim=0)

    # Find the tensor closest to the average
    closest_tensor_index = torch.argmin(torch.norm(tensor_array - average_tensor, dim=(1, 2)))
    diverse_subset = [tensor_array[closest_tensor_index]]
    diverse_names = [names[closest_tensor_index]]
    
    # Mask to keep track of selected tensors
    selected_mask = torch.zeros(len(tensor_array), dtype=torch.bool)
    selected_mask[closest_tensor_index] = True
    #print(f"Most average: {names[closest_tensor_index]}")

    # Iteratively add tensors
    for _ in range(1, N):
        # Calculate the average tensor of the current subset
        subset_average = torch.mean(torch.stack(diverse_subset), dim=0)

        # Find the tensor that is furthest from the current subset average
        distances = torch.norm(tensor_array - subset_average, dim=(1, 2))
        distances[selected_mask] = float('-inf')  # Ignore already selected tensors
        furthest_tensor_index = torch.argmax(distances)
        
        # Add the furthest tensor to the subset and update the mask
        diverse_subset.append(tensor_array[furthest_tensor_index])
        selected_mask[furthest_tensor_index] = True
        diverse_names.append(names[furthest_tensor_index])
    
    print(f"Most diverse {len(diverse_subset)} samples: {diverse_names}")
    
    return torch.stack(diverse_subset)



from SampleCategory import *

# Utility to help categorise samples:
def display_sample_categories():
    print("\n\n\nTesting sample category labbling\n")
    stfts, file_names = load_STFTs()
    others = []
    
    for name in file_names:
        category = infer_sample_category(name)
        #print(f"{category:>20}: {name}")
        if category == "No Category":
            for term in split_text_into_words(name):
                if not ignore_term(term):
                    others.append(term)

    print("Common unmatched terms:")
    display_top_words(others, 0.0)
    
    infer_sample_categories(file_names)
    

#if __name__ == '__main__':
#    display_sample_categories()

