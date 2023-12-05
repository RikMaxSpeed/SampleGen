import re
from collections import Counter

from MakeSTFTs import *


def add_tensors(tensor1, tensor2):
    # Ensure the first dimensions are equal
    assert tensor1.size(0) == tensor2.size(0), "The first dimensions are not equal"

    # Determine the maximum size for the second dimension
    max_size_second_dim = max(tensor1.size(1), tensor2.size(1))

    # Create new tensors with the first dimension same as the originals, and 
    # the second dimension equal to the maximum size, initialized with zeros
    padded_tensor1 = torch.zeros(tensor1.size(0), max_size_second_dim, dtype=tensor1.dtype)
    padded_tensor2 = torch.zeros(tensor2.size(0), max_size_second_dim, dtype=tensor2.dtype)

    # Copy the original tensor values to the new tensors
    padded_tensor1[:, :tensor1.size(1)] = tensor1
    padded_tensor2[:, :tensor2.size(1)] = tensor2

    # Add and return the result
    return padded_tensor1 + padded_tensor2


def get_sorted_words_by_frequency(strings):
    # Use a regular expression to split strings into words using non-alpha characters as delimiters
    words = [word.lower() for s in strings for word in re.split(r'[^a-zA-Z]+', s) if word]
    
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Sort words by frequency in decreasing order
    sorted_words = sorted(word_counts.keys(), key=lambda x: (-word_counts[x], x))
    
    return sorted_words


def display_common_words():
    stfts, file_names = load_STFTs()

    frequent_words = get_sorted_words_by_frequency(file_names)
    use = len(frequent_words) // 4
    print("Top {} words:".format(use))
    for i in range(use):
        print("#{}: {}".format(i+1, frequent_words[i]))


keywords = "guitar bass pad oh saw strings plect string pluck voice bell choir piano bandura epiano mini sweep vox buzz banjo brass chime harp human sync lead organ sitar vocoder water bowed dulcimer ensemble mellotron sine"

keywords = re.split(' ', keywords)


def random_stft(file_names, count):
    assert(count <= len(file_names))
    
    while True:
        i = np.random.randint(count)
        f = file_names[i].lower()
        
        for k in keywords:
            if k in f:
                return i


def get_training_stfts(total = None):
    stfts, file_names = load_STFTs()
    count = len(stfts)
    
    if total is None:
        total = count
        
    if count >= total: # We could shuffle, but I'll keep it static for now.
        return stfts[:total], file_names[:total]
    
    # Augment
    add = total - count
    
    print("Augmenting using {} random mixes".format(add))
    
    for i in range(add):
        r1 = random_stft(file_names, count)
        r2 = random_stft(file_names, count)
        
        stft = add_tensors(stfts[r1], stfts[r2]) / 2
        name = "mix " + file_names[r1][:-4] + " & " + file_names[r2][:-4]
        stfts.append(stft)
        file_names.append(name)

    return stfts, file_names
