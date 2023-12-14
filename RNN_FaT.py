import torch
import torch.nn as nn
import torch.nn.functional as F
from MakeSTFTs import *
from ModelUtils import rnn_size, periodically_display_2D_output
from VariationalAutoEncoder import reconstruction_loss, VariationalAutoEncoder
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters
from Debug import debug

# Adds a time-dimsension to the original frequency RNN

##########################################################################################

class RNNFreqAndTime(nn.Module): # no VAE
                
    @staticmethod
    def approx_trainable_parameters(freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth):
    
        encode_freq = rnn_size(freq_buckets, freq_size, freq_depth)
        decode_freq = rnn_size(freq_size, freq_buckets, freq_depth)

        encode_time = rnn_size(sequence_length, time_size, time_depth)
        decode_time = rnn_size(time_size, sequence_length, time_depth)

        return encode_freq + decode_freq + encode_time + decode_time

    def __init__(self, freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth, dropout):
        super(RNNFreqAndTime, self).__init__()
        
        freq_size = int(freq_size) # RNN objects to int64.
        self.sequence_length = sequence_length
        self.freq_size = freq_size
        self.time_size = time_size
        
        # RNN: input = (batch, sequence, features)
        self.encode_freq = torch.nn.RNN(freq_buckets, freq_size, num_layers = freq_depth, batch_first = True, dropout = dropout)
        self.decode_freq = torch.nn.RNN(freq_size, freq_buckets, num_layers = freq_depth, batch_first = True, dropout = dropout)

        self.encode_time = torch.nn.RNN(sequence_length, time_size, num_layers = time_depth, batch_first = True, dropout = dropout)
        self.decode_time = torch.nn.RNN(time_size, sequence_length, num_layers = time_depth, batch_first = True, dropout = dropout)

        compression = (freq_buckets * sequence_length) / (freq_size * time_size)
        
        print(f"RNN Frequency & Time {count_trainable_parameters(self):,} parameters, compression={compression:.1f}")
        
    def encode(self, x):
        batch_size = x.size(0)
        
        x = x.transpose(2, 1) # Swap from [batch, stft, seq] to [batch, seq, stft] which is what the RNN expects
        x, _ = self.encode_freq(x) # also returns the last internal state
        
        x = x.transpose(2, 1)
        x, _ = self.encode_time(x)

        periodically_display_2D_output(x)

        #x = torch.tanh(x) # Unnecessary as the RNN uses TanH by default.

        x = x.flatten(-2)
        
        return x


    def decode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.freq_size, self.time_size)
        
        x, _ = self.decode_time(x)
        
        x = x.transpose(2, 1)
        x, _ = self.decode_freq(x)
        
        x = x.transpose(2, 1)
        
        return x


    def forward(self, inputs):
        latent = self.encode(inputs)
        outputs = self.decode(latent)
        return outputs
              
              
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs


if __name__ == '__main__':
    from MakeSTFTs import *
    
    freq_size  = 30
    freq_depth = 2
    time_size  = 15
    time_depth = 3
    dropout = 0
    
    approx_size = RNNFreqAndTime.approx_trainable_parameters(freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth)
    print(f"approx_size={approx_size:,}")
    
    fat = RNNFreqAndTime(freq_buckets, sequence_length, freq_size, freq_depth, time_size, time_depth, dropout)
    real_size = count_trainable_parameters(fat)
    print(f"real_size={real_size:,}")
    
    assert(approx_size == real_size)

    batch = 3
    inputs = torch.rand((batch, freq_buckets, sequence_length))
    debug("inputs", inputs)

    e = fat.encode(inputs)
    debug("encode", e)
    
    assert(e.size(0) == batch)
    assert(e.size(1) == freq_size * time_size)
    
    d = fat.decode(e)
    debug("decode", d)
    
    f = fat.forward(inputs)
    debug("forward", f)
    
    l, o = fat.forward_loss(inputs)
    debug("loss", l)
    debug("outputs", o)
    
    
