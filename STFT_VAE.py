from VariationalAutoEncoder import *

# Simply create a massive VAE with the entire spectogram as input
# This does work but requires >100M parameters and has a lot of noise in the results.

class STFTVariationalAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(sequence_length, stft_buckets, sizes):
        sizes = [sequence_length * stft_buckets] + sizes
        return VariationalAutoEncoder.approx_trainable_parameters(sizes)


    def __init__(self, sequence_length, stft_buckets, sizes, activation_fn):
        super(STFTVariationalAutoEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.stft_buckets = stft_buckets
        sizes = [sequence_length * stft_buckets] + sizes
        self.vae = VariationalAutoEncoder(sizes, activation_fn)
        
        
    def encode(self, x, randomize):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.vae.encode(x)
        
        if randomize:
            return self.vae.reparameterize(mu, logvar) # for generation
        else:
            return mu # for precise reproduction
        
        
    def decode(self, x):
        x = self.vae.decode(x)
        print("Check this code!")
        x = torch.nn.Tanh(x) # Restrict to [-1, 1]
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x
        
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x, mu, logvar = self.vae.forward(x)
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x, mu, logvar


    def loss_function(self, inputs, outputs, mu, logvar):
        return self.vae.loss_function(inputs, outputs, mu, logvar)


    def forward_loss(self, inputs):
        outputs, mus, logvars = self.forward(inputs, True)
        loss = self.loss_function(inputs, outputs, mus, logvars)
        return loss, outputs
