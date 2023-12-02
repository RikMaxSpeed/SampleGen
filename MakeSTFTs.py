import os
import numpy as np
import librosa
import torch
import pickle
from AudioUtils import *
import time
import sys

# These global variables are re-used in Train.py
# Would be better to create a class to store these
sample_rate = 44100
stft_buckets = 1024
stft_hop = int(1024 * 3 / 4)
print("Using sample rate={} Hz, FFT={} buckets, hop={} samples".format(sample_rate, stft_buckets, stft_hop))


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


def compute_stft_from_file(filename):
    """Computes the STFT of the audio file and returns it as a tensor"""
    sr, stft = compute_stft_for_file(filename, 2*stft_buckets, stft_hop)
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


def save_to_file(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_from_file(filename):
    with open(filename, 'rb') as file:
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
