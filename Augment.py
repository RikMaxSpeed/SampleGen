from MakeSTFTs import *
import re


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



# Augment the dataset by mixing two random samples:
keywords = "guitar bass pad oh saw strings plect string pluck voice bell choir piano bandura epiano mini sweep vox buzz banjo brass chime harp human sync lead organ sitar vocoder water bowed dulcimer ensemble mellotron sine"

keywords = re.split(' ', keywords)

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


def random_stft(file_names, count):
    assert(count <= len(file_names))
    
    while True:
        i = np.random.randint(count)
        f = file_names[i].lower()
        
        for k in keywords:
            if k in f:
                return i


# TODO: apply augmentation to the training set only otherwise we have cross-contamination
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
