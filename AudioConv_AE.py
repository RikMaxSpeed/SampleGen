import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from Debug import debug
from ModelUtils import conv1d_size, periodically_display_2D_output, model_output_shape_and_size, conv1d_output_size
from VariationalAutoEncoder import reconstruction_loss, basic_reconstruction_loss, CombinedVAE
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters
from Device import device


# Amazingly, it turns out we're better off using the simple MSE rather than comparing STFTs!!!
# def torch_stft(sample, fft_size = 1024): # verified that 256 is the fastest on my Mac. We're compromising on frequency resolution vs time though...
#     return torch.stft(sample.cpu(), n_fft=fft_size, return_complex=True).abs().to(device)
#
# if __name__ == '__main__':
#     count = 1024
#     sample = torch.rand(count, 85000).to(device)
#     fastest_size = None
#     fastest_elapsed = 1e99
#
#     for n in range(4, 16):
#         fft_size = 2**n
#
#         now = time.time()
#
#         torch_stft(sample, fft_size)
#
#         elapsed = time.time() - now
#
#         print(f"fft_size={fft_size}, count={count} FFTs in {elapsed:.4f} sec")
#         if elapsed < fastest_elapsed:
#             fastest_size = fft_size
#             fastest_elapsed = elapsed
#     print(f"\nFastest FFT size={fastest_size}")



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

        # Add a tanh to the encoder so the VAE only sees numbers between [-1, 1].
        # And similarly to the decoder so the audio doesn't saturate.
        layers.append(torch.nn.Tanh())

        return nn.Sequential(*layers)

    def __init__(self, audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size):
        super(AudioConv_AE, self).__init__()

        self.audio_length = audio_length
        self.kernel_count = kernel_count

        self.encoder = self.make_layers(False, depth, kernel_count, outer_kernel_size, inner_kernel_size)
        self.decoder = self.make_layers(True,  depth, kernel_count, outer_kernel_size, inner_kernel_size)

        #try:
        if True:
            self.encoded_shape, self.encoded_size = model_output_shape_and_size(self.encoder, [audio_length])
            print(f"encoded shape={self.encoded_shape}, size={self.encoded_size}")
            assert self.encoded_shape[0] == self.kernel_count
            assert self.encoded_shape[1] == self.expected_length
            assert self.encoded_size == self.expected_length * self.kernel_count, f"expected encoded_size={self.expected_length * self.kernel_count} but got {encoded_size}"

            decode_shape, decode_size = model_output_shape_and_size(self.decoder, self.encoded_shape)
            #print(f"decoded shape={decode_shape}, size={decode_size}")
        # except Exception as e:
        #     print(f"Model doesn't work: {e}")
        #     self.compression = 0
        # except BaseException as e:
        #     print(f"Model is broken! {e}")
        #     self.compression = 0
        #     return

        self.compression = audio_length / self.encoded_size
        print(f"AudioConv_AE {count_trainable_parameters(self):,} parameters, compression={self.compression:.1f}")

    def encode(self, x):
        #debug("encode.x", x)
        batch_size = x.size(0)
        assert(x.size(1) == self.audio_length)
        x = x.view(batch_size, 1, self.audio_length)
        #debug("x.view", x)
        hiddens = self.encoder(x)
        #debug("hiddens", hiddens)

        #hiddens = hiddens.transpose(2, 1) # convert to [batch, time, features]
        periodically_display_2D_output(hiddens)

        #encoded = hiddens.flatten(-2)
        #debug("encoded", encoded)
        return hiddens

    def decode(self, x):
        #debug("decode.x", x)
        batch_size = x.size(0)
        #hiddens = x.view(batch_size, self.kernel_count, -1)
        #x = x.transpose(2, 1) # convert to [batch, features, time] for the ConvTranspose1D
        #debug("hiddens", hiddens)
        decoded = self.decoder(x)
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
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs

    def stft_loss(self, inputs, outputs):
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


if __name__ == "__main__":
    # audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size
    model = AudioConv_AE(85_000, 2, 25, 80, 70)
    model.float()
    model.to(device)
    audio_length = 85_000
    print(f"encoded_shape={model.encoded_shape}")
    batched_input = torch.randn(7, audio_length).to(device)
    print(f"batched_input.shape={batched_input.shape}")
    loss, batched_output = model.forward_loss(batched_input)
    print(f"loss={loss}, batched_output={batched_output.shape}")
    assert batched_output.shape == batched_input.shape
    print("\nAudioConv_AE is OK!\n\n")

    vae_sizes = [list(model.encoded_shape), 300, 200, 8]
    combined = CombinedVAE(model, vae_sizes)
    combined.float()
    combined.to(device)

    loss, batched_output = combined.forward_loss(batched_input)
    print(f"loss={loss}, batched_output={batched_output.shape}")
    assert batched_output.shape == batched_input.shape
    assert abs(loss - audio_length) < 2000, f"loss={loss} is greater than expected"

    hidden_input = combined.auto_encoder.encode(batched_input)
    debug("hidden_input", hidden_input)

    loss, hidden_output = combined.vae.forward_loss(hidden_input)
    print(f"loss={loss}")
    debug("hidden_output", hidden_output)
    assert hidden_input.shape == hidden_output.shape


    print("\nAudioConv_VAE is OK!\n\n")

