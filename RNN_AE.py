import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils import rnn_size, periodically_display_2D_output
from VariationalAutoEncoder import reconstruction_loss, VariationalAutoEncoder
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters

# Buildling on the experience of the rnnVAE, we now try using an RNN to extract the time hiddens of the signal.
# We then combine that with the VAE to massively reduce the feature space.

##########################################################################################

class RNNAutoEncoder(nn.Module): # no VAE
                
    @staticmethod
    def approx_trainable_parameters(freq_buckets, hidden_size, encode_depth, decode_depth):
        encode = rnn_size(freq_buckets, hidden_size, encode_depth)
        decode = rnn_size(hidden_size, freq_buckets, decode_depth)
        return encode + decode

    def __init__(self, freq_buckets, sequence_length, hidden_size, encode_depth, decode_depth, dropout):
        super(RNNAutoEncoder, self).__init__()
        
        hidden_size = int(hidden_size) # RNN objects to int64.
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # RNN: input = (batch, sequence, features)
        self.encoder = torch.nn.RNN(freq_buckets, hidden_size, num_layers = encode_depth, batch_first = True, dropout = dropout)
        self.decoder = torch.nn.RNN(hidden_size, freq_buckets, num_layers = decode_depth, batch_first = True, dropout = dropout)
        # Important: in the rnnVAE I gave the model the previous and the current frame at each time step + the time itself.
        # In this version I'm only giving the model the individual time-steps... I don't know whether this will work.
        
        print(f"RNNAutoEncoder {count_trainable_parameters(self):,} parameters, compression={freq_buckets/hidden_size:.1f}")        
        
    def encode(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1) # Swap from [batch, stft, seq] to [batch, seq, stft] which is what the RNN expects
        hiddens, _ = self.encoder(x) # also returns the last internal state
        
        # Check that the encoded layer is between -1 and 1
        #print(f"RNNAutoEncoder: hidden={hiddens.shape}, min={hiddens.min():.3f}, max={hiddens.max():.3f}")
        #assert(hiddens.abs().max() <= 1)
        #hiddens = torch.sigmoid(hiddens) # do we really need this?
        
        periodically_display_2D_output(hiddens)

        hiddens = hiddens.flatten(-2)
        
        return hiddens


    def decode(self, x):
        batch_size = x.size(0)
        hiddens = x.view(batch_size, self.sequence_length, self.hidden_size)#.to(device)
        reconstructed, _ = self.decoder(hiddens) # Note: RNN uses Tanh by default so all our outputs will be constrained to [-1, 1]
        reconstructed = reconstructed.transpose(2, 1)
        return reconstructed


    def forward(self, inputs):
        hiddens = self.encode(inputs)
        outputs = self.decode(hiddens)
        return outputs
              
              
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs
