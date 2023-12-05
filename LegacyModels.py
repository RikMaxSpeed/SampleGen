# Ignore: old code from models I tried previously




###############################################################################################################################################
# I've had little success using LSTM, GRU or RNN...

class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, sequence_length, rnn_size, rnn_layers, mlp_layers, activation):
        super(AutoEncoderRNN, self).__init__()
        
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.activation = activation
        
        # Encoder
        #self.encode_rnn = nn.LSTM(input_size, rnn_size, num_layers=rnn_layers, batch_first=True)
        self.encode_rnn = nn.GRU(input_size, rnn_size, num_layers=rnn_layers, batch_first=True)
        #rnn = nn.RNN(input_size, rnn_size, num_layers=rnn_layers, batch_first=True, nonlinearity='relu')
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #self.kernel_size = rnn_size//4
        #self.conv = nn.Conv1D(input_size, rnn_size, kernel_size)
        
        sizes = [rnn_size * sequence_length] + mlp_layers
        print("encoder=", sizes)
        self.encode_linears = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])

        # Decoder
        sizes.reverse()
        print("decoder=", sizes)
        self.decode_linears = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        self.decode_rnn = nn.GRU(rnn_size, input_size, num_layers=rnn_layers, batch_first=True)


    def encode(self, x):
        out, _ = self.encode_rnn(x)
        x = out.reshape(out.size(0), -1)
        
        for linear in self.encode_linears:
            x = self.activation(linear(x))
            
        return x


    def decode(self, x):
        for linear in self.decode_linears:
            x = self.activation(linear(x))

        x = x.reshape(x.size(0), self.sequence_length, self.rnn_size)
        
        x, _ = self.decode_rnn(x)
        
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x





###############################################################################################################################################
# Simpler alternative with no LSTM
# This worked for small inputs, but not for the full data-set.
# ie: 100 training samples of 0.5 seconds, up to 2 seconds. But was impossible to scale to 1000 samples.

class AutoEncoderMLP(nn.Module):
    def __init__(self, N, M, D, hidden_dims):
        super(AutoEncoderMLP, self).__init__()

        self.N = N
        self.M = M
        self.activation = nn.ReLU() #nn.ReLU() nn.Tanh() nn.Sigmoid()
        
        # Encoder
        sizes = [N*M] + hidden_dims + [D]
        print("encode={}".format(sizes))
        self.encode_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])

        # Decoder
        sizes.reverse();
        print("decode={}".format(sizes))
        self.decode_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        

    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        
        for linear in self.encode_layers:
            x = self.activation(linear(x))
        
        return x

    def decode(self, x):
        for linear in self.decode_layers:
            x = self.activation(linear(x))
                
        x = x.reshape(x.size(0), self.N, self.M)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded





class AutoEncoder(nn.Module):
    def __init__(self, input_channels=1, image_size=(1024, 400), encoder_channels=[16, 32, 64, 128], D=128, kernel_size=3, padding=1, stride=2, pool_size=2):
        super(AutoEncoder, self).__init__()

        # Encoder
        layers = []
        in_channels = input_channels
        for out_channels in encoder_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.MaxPool2d(pool_size, pool_size)
            ])
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        
        # Calculate output shape after encoder
        H, W = image_size
        for _ in encoder_channels:
            H = ((H - kernel_size + 2*padding) // stride + 1) // pool_size
            W = ((W - kernel_size + 2*padding) // stride + 1) // pool_size

        self.enc_out_shape = (in_channels, H, W)
        
        self.fc1 = nn.Linear(in_channels*H*W, D)
        self.fc2 = nn.Linear(D, in_channels*H*W)

        # Decoder
        layers = []
        decoder_channels = list(reversed(encoder_channels[:-1])) + [input_channels]  # reverse and append the input channel
        for out_channels in decoder_channels:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU()
            ])
            in_channels = out_channels
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = F.relu(self.fc2(x))
        x = x.reshape(x.size(0), *self.enc_out_shape)
        x = self.decoder(x)
        
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x



###############################################################################################################################################
# This works but the model is huge and slow to train.
# It also overfits the training data massively in order to get good results.

class SymmetricAutoEncoder(nn.Module):
    def __init__(self, sizes, activation_fn):
        super(SymmetricAutoEncoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.decoder_layers = nn.ModuleList([nn.Linear(sizes[i+1], sizes[i]) for i in reversed(range(len(sizes)-1))])
        self.activation_fn = activation_fn

    def encode(self, x):
        idx = 0
        for layer in self.encoder_layers:
            idx += 1
            x = self.activation_fn(layer(x))
        
        return x
    
    def decode(self, z):
        for idx, layer in enumerate(self.decoder_layers):
            z = layer(z)  # Using the decoder layers
            if idx < len(self.decoder_layers) - 1:  # Don't apply activation to the last layer
                z = self.activation_fn(z)
                
        return z

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x



class STFTAutoEncoder(nn.Module):
    def __init__(self, sequence_length, stft_buckets, sizes, activation_fn):
        super(STFTAutoEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.stft_buckets = stft_buckets
        sizes = [sequence_length * stft_buckets] + sizes
        print("STFTAutoEncoder: sequence_length={}, stft_buckets={}, sizes={}, activation_fn={}".format(sequence_length, stft_buckets, sizes, activation_fn.__class__))
        self.AutoEncoder = SymmetricAutoEncoder(sizes, activation_fn)
        
    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        return self.AutoEncoder.encode(x)
        
    def decode(self, x):
        x = self.AutoEncoder.decode(x)
        x = x.reshape(x.size(0), self.sequence_length, self.stft_buckets)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x




##########################################################################################
# CNN/RNN Auto-encoder with no VAE
#

class HybridCNNAutoEncoder(nn.Module):
    @staticmethod
    def approx_trainable_parameters(stft_buckets, seq_length, kernel_count, kernel_size, rnn_hidden_size):
        # Parameters in the 1D Conv layer
        conv_params = (stft_buckets * kernel_count * kernel_count) + kernel_count  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        # Parameters in the GRU layer - Encoder
        rnn_input_size = kernel_count * seq_length
        encoder_rnn_params = 3 * (rnn_hidden_size**2 + rnn_hidden_size * seq_length + rnn_hidden_size)

        # Parameters in the GRU layer - Decoder (similar to encoder)
        decoder_rnn_params = encoder_rnn_params #3 * (rnn_hidden_size**2 + rnn_hidden_size * rnn_input_size + rnn_hidden_size)

        # Parameters in the 1D Transposed Conv layer
        deconv_params = (kernel_count * stft_buckets * kernel_count) + stft_buckets  # (in_channels * out_channels * kernel_size) + out_channels (for bias)

        total_params = conv_params + encoder_rnn_params + decoder_rnn_params + deconv_params
        return total_params


    def __init__(self, stft_buckets, seq_length, kernel_count, kernel_size, rnn_hidden_size):
        super(HybridCNNAutoEncoder, self).__init__()
        self.stft_buckets = stft_buckets
        self.seq_length = seq_length
        self.rnn_hidden_size = rnn_hidden_size

        # Simulate
        batch=7
        x = torch.rand(batch, stft_buckets, seq_length)
    
        # Encoder
        conv_pad = 0
        conv_stride = 1
        x = x.unsqueeze(1)
        print(f"x.unsqueeze={x.shape}")
        self.encoder_conv = nn.Conv2d(in_channels=1, out_channels=kernel_count, kernel_size=(1, kernel_size), stride=conv_stride, padding=conv_pad)
        x = get_output_for_layer("encoder_conv", self.encoder_conv, x)
    
        stft_kernel_results = x.size(1) * x.size(2)
        x = x.reshape(x.size(0), x.size(3), stft_kernel_results)
        print(f"x.reshape={x.shape}")
        
        self.encoder_rnn = nn.GRU(input_size=stft_kernel_results, hidden_size=rnn_hidden_size, batch_first=True, num_layers=1, dropout=0) # more hyper-parameters!
        x = get_output_for_layer("encoder_rnn", self.encoder_rnn, x)
        
        # Latent
        self.latent_size = x.size(1) * x.size(2)
        x = x.reshape(x.size(0), self.latent_size)
        print(f"latent={x.shape}")
        
        # Decoder
        x = torch.rand(batch, self.latent_size)
        x = x.view(x.shape[0], -1, rnn_hidden_size)
        self.decoder_rnn = nn.GRU(input_size=rnn_hidden_size, hidden_size=stft_kernel_results, batch_first=True)
        x = get_output_for_layer("decoder_rnn", self.decoder_rnn, x)
        x = x.unsqueeze(1)
        print(f"x.unsqueeze={x.shape}")
        self.decoder_conv = nn.ConvTranspose2d(in_channels=1, out_channels=stft_buckets, kernel_size=(1, kernel_size), stride=conv_stride, padding=conv_pad)
        x = get_output_for_layer("decoder_conv", self.decoder_conv, x)
        print("model created successfully!\n\n")
        
        
    def encode(self, x):
        x = self.encoder_conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), x.size(2), -1)
        x, _ = self.encoder_rnn(x)
        x = x.reshape(x.size(0), self.latent_size)
        return x

    def decode(self, x):
        assert(x.shape[1] == self.latent_size)
        x = x.view(x.shape[0], -1, self.rnn_hidden_size)
        x, _ = self.decoder_rnn(x)
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.decoder_conv(x)
        assert(x.shape[1] == self.stft_buckets)
        x = x[:, :, :self.seq_length] # truncate to the expected sequence length
        assert(x.shape[2] == self.seq_length)
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
        
        
    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        return reconstruction_loss(outputs, inputs), outputs
