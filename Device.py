import torch
import os

# Default to CPU if an MPS function is not implemented. Required for PyTorch.stft
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Turns out we don't need this!

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# On the M3 MacBook Pro, MPS is at least 10x faster than CPU
# However I have bumped into at least 1 bug where MPS simply crashes and aborts the process.
# The only work-around is to use 'cpu'
device = 'cpu'

print(f"\n*** Using device={device} ***\n")

