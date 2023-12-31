import torch

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# On my M3 Mac, MPS is at least 10x faster than CPU
# However I have bumped into 1 bug where MPS simply aborts the process.
# The only work-around is to use 'cpu'
#device = 'cpu'

print("Using device={}".format(device))
