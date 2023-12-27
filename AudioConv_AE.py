import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from Debug import debug
from ModelUtils import conv1d_size, periodically_display_2D_output, model_output_shape_and_size, conv1d_output_size
from VariationalAutoEncoder import reconstruction_loss, basic_reconstruction_loss
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters
from Device import device


def torch_stft(sample, fft_size = 1024): # verified that 256 is the fastest on my Mac. We're compromising on frequency resolution vs time though...
    return torch.stft(sample.cpu(), n_fft=fft_size, return_complex=True).abs().to(device)

if __name__ == '__main__':
    count = 1024
    sample = torch.rand(count, 85000).to(device)
    fastest_size = None
    fastest_elapsed = 1e99

    for n in range(4, 16):
        fft_size = 2**n

        now = time.time()

        torch_stft(sample, fft_size)

        elapsed = time.time() - now

        print(f"fft_size={fft_size}, count={count} FFTs in {elapsed:.4f} sec")
        if elapsed < fastest_elapsed:
            fastest_size = fft_size
            fastest_elapsed = elapsed
    print(f"\nFastest FFT size={fastest_size}")

# def sample_hash_key(sample):
#     assert sample.ndimension() == 1, f"Expected a 1D tensor, got {sample.shape}"
#     key = 0
#     for i in range(0, sample.shape[0], 100):
#         key += sample[i]
#     return str(key)
#
# def cached_batched_torch_stft():
#     poo()

class AudioConv_AE(nn.Module):  # no VAE

    @staticmethod
    def approx_trainable_parameters(depth, kernel_count, outer_kernel_size, inner_kernel_size):
        encode = conv1d_size(1, kernel_count, outer_kernel_size) \
                 + (depth - 1) * conv1d_size(kernel_count, kernel_count, inner_kernel_size)

        decode = conv1d_size(kernel_count, 1, outer_kernel_size) \
                 + (depth - 1) * conv1d_size(kernel_count, kernel_count, inner_kernel_size)

        return encode + decode

    def make_layers(self, is_decoder, depth, kernel_count, outer_kernel_size, inner_kernel_size):
        assert(depth >= 1)

        length = self.audio_length

        layers = []
        # Outer layer
        stride = max(outer_kernel_size // 8, 1)
        if is_decoder:
            layers.append(torch.nn.ConvTranspose1d(kernel_count, 1, outer_kernel_size, stride=stride))
        else:
            layers.append(torch.nn.Conv1d(1, kernel_count, outer_kernel_size, stride=stride))

        length = conv1d_output_size(length, outer_kernel_size, stride)

        for i in range(1, depth):
            #stride = max(2, inner_kernel_size - i)
            stride = max(2, inner_kernel_size // 2 - i)

            length = conv1d_output_size(length, inner_kernel_size, stride)
            if length <= 4: # over-compressing
                break

            if is_decoder:
                layers.append(torch.nn.ConvTranspose1d(kernel_count, kernel_count, inner_kernel_size, stride=stride))
            else:
                layers.append(torch.nn.Conv1d(kernel_count, kernel_count, inner_kernel_size, stride=stride))

            #layers.append(torch.nn.LeakyReLU())


        if is_decoder:
            layers.reverse()
            print(f"Expect final sequence length={length}")
            self.expected_length = length
        else:
            layers.append(torch.nn.Tanh())

        return nn.Sequential(*layers)

    def __init__(self, audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size):
        super(AudioConv_AE, self).__init__()

        self.audio_length = audio_length
        self.kernel_count = kernel_count

        self.encoder = self.make_layers(False, depth, kernel_count, outer_kernel_size, inner_kernel_size)
        self.decoder = self.make_layers(True,  depth, kernel_count, outer_kernel_size, inner_kernel_size)

        try:
            encoded_shape, encoded_size = model_output_shape_and_size(self.encoder, [1, audio_length])
            #print(f"encoded shape={encoded_shape}, size={encoded_size}")
            assert encoded_shape[1] == self.expected_length
            self.encoded_size = encoded_size # required for the CombinedVAE

            decode_shape, decode_size = model_output_shape_and_size(self.decoder, encoded_shape)
            #print(f"decoded shape={decode_shape}, size={decode_size}")
        except BaseException as e:
            print(f"Model doesn't work: {e}")
            self.compression = 0
            return

        self.compression = audio_length / encoded_size
        print(f"AudioConv_AE {count_trainable_parameters(self):,} parameters, compression={self.compression:.1f}")

    def encode(self, x):
        #debug("encode.x", x)
        batch_size = x.size(0)
        assert(x.size(1) == self.audio_length)
        x = x.view(batch_size, 1, self.audio_length)
        #debug("x.view", x)
        hiddens = self.encoder(x)
        #debug("hiddens", hiddens)

        periodically_display_2D_output(hiddens)

        encoded = hiddens.flatten(-2)
        #debug("encoded", encoded)
        return encoded

    def decode(self, x):
        #debug("decode.x", x)
        batch_size = x.size(0)
        hiddens = x.view(batch_size, self.kernel_count, -1)
        #debug("hiddens", hiddens)
        decoded = self.decoder(hiddens)
        #debug("decoded", decoded)
        decoded = decoded.view(batch_size, -1)
        #debug("decoded.view", decoded)
        len = decoded.size(1)
        missing = self.audio_length - len
        assert(missing >= 0)
        audio = F.pad(decoded, (0, missing))
        #debug("audio", audio)
        return audio

    def forward(self, inputs):
        hiddens = self.encode(inputs)
        outputs = self.decode(hiddens)
        return outputs

    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = self.stft_loss(inputs, outputs)
        return loss, outputs

    def stft_loss(self, inputs, outputs):
        # Naïve version:
        return basic_reconstruction_loss(inputs, outputs)

        # Compare the STFT instead, fortunately PyTorch provides a differentiable STFT
        fft_size = 2048

        # torch.stft is not supported on MPS so we have to move things back to the CPU
        #now = time.time()

        # Cache the input STFT for speed
        # key = sample_hash_key(inputs)
        # inputs = self.cached_stfts.get(key)
        # if inputs is None:
        #     print(f"cache miss for {key}")
        #     self.cached_stfts[key] = inputs

        inputs  = torch_stft(inputs)
        outputs = torch_stft(outputs)
        max_amp = inputs.abs().max()

        return basic_reconstruction_loss(inputs, outputs)/(max_amp**2)
