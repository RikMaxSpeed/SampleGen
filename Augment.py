from MakeSTFTs import *
from torch.utils.data import TensorDataset


#def add_tensors(tensor1, tensor2):
#    # Ensure the first dimensions are equal
#    assert tensor1.size(0) == tensor2.size(0), "The first dimensions are not equal"
#
#    # Determine the maximum size for the second dimension
#    max_size_second_dim = max(tensor1.size(1), tensor2.size(1))
#
#    # Create new tensors with the first dimension same as the originals, and 
#    # the second dimension equal to the maximum size, initialized with zeros
#    padded_tensor1 = torch.zeros(tensor1.size(0), max_size_second_dim, dtype=tensor1.dtype)
#    padded_tensor2 = torch.zeros(tensor2.size(0), max_size_second_dim, dtype=tensor2.dtype)
#
#    # Copy the original tensor values to the new tensors
#    padded_tensor1[:, :tensor1.size(1)] = tensor1
#    padded_tensor2[:, :tensor2.size(1)] = tensor2
#
#    # Add and return the result
#    return padded_tensor1 + padded_tensor


def get_training_stfts(total = None):
    stfts, file_names = load_STFTs()
    count = len(stfts)
    
    if total is None or count >= total:
        stfts, file_names
        
    return stfts[:total], file_names[:total] # We could shuffle, but I'll keep it static for now.

    
def augment_stfts(stfts, total):
    add = total - len(stfts)
    
    if add > 0 :
        print("Augmenting training dataset using {} random mixes".format(add))
        new_stfts = []
        # We could use other operations: time stretching, pitch-shifting by an octave up or down, truncating etc.
        # But we dont' want to disturb the structure that the model is trying to learn, so keep it simple for now.
        original_size = len(stfts)
        for i in range(add):
            r1 = np.random.randint(original_size)
            r2 = np.random.randint(original_size)
            if r1 == r2:
                r2 = (1 + r2) % original_size
            
            assert(r1 != r2)
            assert(r1 < original_size)
            assert(r2 < original_size)
            
            mix = np.random.uniform(0.3, 0.7)

            stft = stfts[r1] * mix +  stfts[r2] * (1 - mix)
            
            new_stfts.append(stft)
        
        
        stfts = torch.utils.data.ConcatDataset([stfts, torch.stack(new_stfts)])
        assert(len(stfts) == total)
        
    return stfts
