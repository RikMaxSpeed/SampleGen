import os
import numpy as np
import librosa
import torch
import pickle
from AudioUtils import *
import time
import sys

# Bunch of nasty global variables...
sample_rate = 44100
stft_buckets = 1024
stft_hop = int(1024 * 3 / 4)

sample_duration = 2.0 # seconds
sequence_length = int(sample_duration * sample_rate / stft_hop)
input_size = stft_buckets * sequence_length

print("Using sample rate={} Hz, FFT={} buckets, hop={} samples, duration={:.1f} sec = {:,} time steps".format(sample_rate, stft_buckets, stft_hop, sample_duration, sequence_length))



def lowest_frequency(stft, sample_rate):
    #debug("lowest_frequency.stft", stft)
    
    amplitudes = abs(stft.sum(axis=1))
    amplitudes[0] = 0 # Remove the constant offset
    
    buckets = len(stft)
    if buckets %2 == 1:
        buckets -= 1
    
    max_amp = amplitudes.max()
    min_amp = max_amp / 32
    for i in range(len(amplitudes)):
        if amplitudes[i] > min_amp:
            return i * sample_rate / (2 * buckets)
            
    print(amplitudes)
    raise Exception("This should never happen!")


def compute_stft_from_file(file_name):
    """Computes the STFT of the audio file and returns it as a tensor"""
    sr, stft = compute_stft_for_file(file_name, 2*stft_buckets, stft_hop)
    #debug("compute_stft_from_file.stft", stft)
    assert(stft.shape[0] == stft_buckets + 1)
    
    low_hz = lowest_frequency(stft, sr)

    return sr, torch.tensor(stft), low_hz


def display_time(start, count, unit):
        elapsed = time.time() - start

        if elapsed > count:
            print("processed {} {}s in {:.1f} sec = {:.1f} sec/{}".format(count, unit, elapsed, elapsed/count, unit))
        else:
            print("processed {} {}s in {:.1f} sec = {:.1f} {}/sec".format(count, unit, elapsed, count/elapsed, unit))


def gather_stfts_from_directory(directory, notes, requiredSR):
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
                            sr, stft_tensor, low_hz = compute_stft_from_file(filepath)
                            
                            if sr != requiredSR:
                                print("Skipping {}: sample rate={} Hz".format(filepath, sr))
                            elif low_hz > c4hz *1.7:
                                print("Skipping {}: lowest frequency={:.1f} Hz".format(filepath, low_hz))
                            else:
                                #debug("stft_tensor", stft_tensor)
                                stft_tensors.append(stft_tensor)
                                file_names.append(file)
                                print("#{}: {}".format(len(stft_tensors)+1, file))
                                    
                        except AssertionError as e:
                            print("Assertion failed in file {}: {}".format(filepath, e))
                            sys.exit()
                            
                        except Exception as e:
                            print("Error reading file {}: {}".format(filepath, e))
    
    print("\n\nDone!!")
    display_time(start, len(stft_tensors), "file")

    return stft_tensors, file_names


def save_to_file(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def load_from_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


stft_file = "STFTs.pkl"


def make_STFTs():
    #notes = ["C5", "C4", "C3", "C2"]
    notes = ["C3", "C4"] # There's confusion over what C4 means, so we have a frequency check instead
    stft_list, file_names = gather_stfts_from_directory("../WaveFiles", notes, 44100)
    save_to_file((stft_list, file_names), stft_file)


def load_STFTs():
    stfts, file_names = load_from_file(stft_file)
    print("Loaded {} STFTs from {}".format(len(stfts), stft_file))
    return stfts, file_names




def adjust_stft_length(stft_tensor, target_length):
    
    assert(stft_buckets <= stft_tensor.shape[0] <= stft_buckets + 1)
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


def convert_stft_to_input(stft):
    assert(len(stft.shape) == 2)
    assert(stft.shape[0] == stft_buckets + 1)
    stft = stft[1:, :]
    assert(stft.shape[0] == stft_buckets)
    
    stft = adjust_stft_length(stft, sequence_length)
    stft = complex_to_mulaw(stft)
    
    return stft.contiguous()


def convert_stfts_to_inputs(stfts):
    return torch.stack([convert_stft_to_input(stft) for stft in stfts]).to(device)


def convert_stft_to_output(stft):
    # Fix dimensions
    amplitudes = stft.squeeze(0)
    
    # Get rid of any silly values...
    amplitudes[amplitudes <= 0] = 0
    
    # Add back the constant bucket = 0
    assert(amplitudes.shape[0] == stft_buckets)
    zeros = torch.zeros(amplitudes.size(1)).unsqueeze(0).to(device)
    amplitudes = torch.cat((zeros, amplitudes), dim=0)
    assert(amplitudes.shape[0] == stft_buckets + 1)
    
    # Convert back to complex with 0 phase
    output_stft = mulaw_to_zero_phase_complex(amplitudes)
    assert(output_stft.shape[0] == stft_buckets + 1)
    output_stft = output_stft.cpu().detach().numpy()

    return output_stft
        
        


def test_stft_conversions(file_name):
    sr, stft = compute_stft_for_file(file_name, 2*stft_buckets, stft_hop)
    debug("stft", stft)
    
    if False: # Generate a synthetic spectrum
        for f in range(stft.shape[0]):
            for t in range(stft.shape[1]):
                stft[f, t] = 75*np.sin(f*t) if f>=2*t and f <= 3*t else 0
    
    plot_stft(file_name, stft, sr, stft_hop)
    stft = np.abs(stft) # because we discard the phases, everything becomes positive amplitudes
    
    tensor = torch.tensor(stft)
    input = convert_stft_to_input(tensor).to(device)
    output = convert_stft_to_output(input)
    
    plot_stft("Resynth " + file_name, output, sr, stft_hop)
    save_and_play_audio_from_stft(output, sr, stft_hop, "Results/resynth-mulaw-" + os.path.basename(file_name), True)

    diff = np.abs(output - stft)
    debug("diff", diff)
    plot_stft("Diff", diff, sr, stft_hop)
    for f in range(stft.shape[0]):
        for t in range(stft.shape[1]):
            d = diff[f, t]
            if d > 0.1:
                print("f={}, t={}, diff={:.3f}, stft={:.3f}, output={:.3f}".format(f, t, d, stft[f, t], output[f, t]))


#test_stft_conversions("Samples/Piano C4 Major 13.wav")
#test_stft_conversions("/Users/Richard/Coding/WaveFiles/FreeWaveSamples/Alesis-S4-Plus-Clean-Gtr-C4.wav")
#sys.exit(1)



from SampleCategory import *

# Utility to help categorise samples:
def display_sample_categories():
    stfts, file_names = load_STFTs()
    others = []
    
    for name in file_names:
        category = infer_sample_category(name)
        #print(f"{category:>20}: {name}")
        if category == "Other":
            for term in split_text_into_words(name):
                if not ignore_term(term):
                    others.append(term)

    print("Common unmatched terms:")
    display_top_words(others, 0.0)
    
    infer_sample_categories(file_names)
    

#display_sample_categories()

