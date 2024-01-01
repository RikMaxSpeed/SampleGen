import torch

the_device = None
def set_device(d):
    global the_device
    the_device = d
    print(f"\n*** Using device={the_device} ***\n")

def get_device():
    return the_device

set_device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


# On the M3 MacBook Pro, MPS is at least 10x faster than CPU
# However I have bumped into at least 1 bug where MPS simply crashes and aborts the process.
# The only work-around is to use 'cpu'
# Bug report filed with Apple.
