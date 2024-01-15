import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from Device import get_device
from Debug import debug
from ModelUtils import conv1d_size, periodically_display_2D_output, model_output_shape_and_size, conv1d_output_size
from VariationalAutoEncoder import reconstruction_loss, basic_reconstruction_loss, CombinedVAE
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters

# Loss function using STFTs:
# However (amazingly) it turns out we're better off using the simple MSE rather than comparing STFTs!!!
def torch_stft(sample, fft_size = 1024): # verified that 256 is the fastest on my Mac. We're compromising on frequency resolution vs time though...
    return torch.stft(sample.cpu(), n_fft=fft_size, return_complex=True).abs().to(get_device())

if __name__ == '__main__' and False: # no longer required
    count = 1024
    sample = torch.rand(count, 85000).to(get_device())
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


class NormalizeWithAmplitudes(nn.Module):
    def __init__(self, temperature=0.1):
        super(NormalizeWithAmplitudes, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        # Using softmax to approximate max and min
        pos_weights = torch.softmax(x / self.temperature, dim=2)
        neg_weights = torch.softmax(-x / self.temperature, dim=2)

        max_approx = torch.sum(pos_weights * x, dim=2)
        min_approx = torch.sum(neg_weights * x, dim=2)


        max_abs_values = torch.where(torch.abs(max_approx) > torch.abs(min_approx), max_approx, min_approx)
        #print(f"Normalised amplitudes: max_abs_values={max_abs_values}")

        max_abs_values = max_abs_values.unsqueeze(2)
        normalized = x / max_abs_values
        return torch.cat((normalized, max_abs_values), dim=2)

class ReverseNormalizeWithAmplitudes(nn.Module):
    def __init__(self):
        super(ReverseNormalizeWithAmplitudes, self).__init__()

    def forward(self, x):
        max_abs_values = x[:, :, -1]
        normalized_data = x[:, :, :-1]
        # Use out-of-place operation for scaling
        result = normalized_data * max_abs_values.unsqueeze(2)
        # Note: no limiting at present...
        return result



if __name__ == '__main__':
    print("\n\nTesting normalisation:")
    batch = 3
    features = 5
    steps = 7

    sample = 10 * torch.rand(batch, features, steps).to(get_device()) - 5
    print(f"sample={sample}\n")

    normalised = NormalizeWithAmplitudes()(sample)
    print(f"normalised={normalised}\n")
    assert normalised.shape[-1] == sample.shape[-1] + 1

    reversed = ReverseNormalizeWithAmplitudes()(normalised)
    print(f"reversed={reversed}\n")

    error = F.mse_loss(sample, reversed, reduction='sum').item()

    print(f"\tbatch={batch}, features={features}, time-steps={steps}, error={error:.8f}")

    assert error < 1e-3, f"Expected a very small error, got {error}"
    print("Normalisation is OK\n\n")


class AudioConv_AE(nn.Module):  # no VAE

    @staticmethod
    def compute_kernel_sizes_and_strides(audio_length, depth, kernel_count, kernel_size, ratio):
        assert ratio > 0 and ratio <= 1, f"invalid ratio={ratio} should be between 0 and 1"

        failed = [], 0, 0
        kernels = []
        strides=[]
        lengths = []
        length = audio_length
        min_length = 1 # this is a bit silly, but let's see what happens...

        for i in range(depth):
            stride = int(kernel_size * ratio) - i
            stride = max(stride, 2)

            kernel = int(kernel_size * (1 - i/depth))
            kernel = max(kernel, 2 * stride)

            if stride <= 1:
                print(f"stride={stride} is too small")
                return failed

            if kernel < 2:
                print(f"kernel={kernel} is too small.")
                return failed

            if stride > kernel:
                print(f"stride={stride} must be less than kernel size={kernel}.")
                return failed

            if kernel >= audio_length:
                print(f"kernel size={kernel} must be less than audio length={audio_length}.")
                return failed

            next_length = conv1d_output_size(length, kernel, stride)

            if next_length < min_length: # over-compressing
                print(f"output length={next_length}, must be at least {min_length}.")
                return failed

            # Valid layer:
            kernels.append(kernel)
            strides.append(stride)
            lengths.append(next_length)

            # Figure out the parameters of the next layer
            stride = int(stride * (1 - ratio))
            stride = max(stride, 2)

            kernel = int(stride/ratio + 1)
            kernel = max(kernel, 2)

            length = next_length

        # Done
        assert len(kernels) == depth
        assert len(strides) == depth

        return kernels, strides, lengths


    @staticmethod
    def approx_trainable_parameters(audio_length, depth, kernel_count, kernel_size, compression):
        kernels, strides, lengths = AudioConv_AE.compute_kernel_sizes_and_strides(audio_length, depth, kernel_count, kernel_size, compression)

        encode = 0
        decode = 0
        for i in range(len(kernels)):
            if i == 0:
                encode += conv1d_size(1, kernel_count, kernels[i])
                decode += conv1d_size(kernel_count, 1, kernels[i])
            else:
                size = conv1d_size(kernel_count, kernel_count, kernels[i])
                encode += size
                decode += size

        return encode + decode

    def make_layers(self, is_decoder, kernel_count, kernels, strides):
        layers = []
        for i in range(len(kernels)):
            size = kernels[i]
            stride = strides[i]
            channels = 1 if i == 0 else kernel_count

            if is_decoder:
                layers.append(torch.nn.ConvTranspose1d(kernel_count, channels, size, stride=stride))
            else:
                layers.append(torch.nn.Conv1d(channels, kernel_count, size, stride=stride))

            #layers.append(torch.nn.LeakyReLU()) # training is markedly worse.

        if is_decoder:
            layers.append(ReverseNormalizeWithAmplitudes())
        else:
            layers.append(NormalizeWithAmplitudes())

        if is_decoder:
            layers.reverse()

        # encoder: needs this otherwise the VAE sees crazy large values.
        # decoder: valid audio is between [-1, 1]
        #layers.append(torch.nn.Hardtanh()) # same as clamp(-1, 1)

        return nn.Sequential(*layers)

    def __init__(self, audio_length, depth, kernel_count, kernel_size, compression):
        super(AudioConv_AE, self).__init__()

        self.compression  = 0 # used to flag the model as invalid
        self.audio_length = audio_length
        self.kernel_count = kernel_count

        kernels, strides, lengths = AudioConv_AE.compute_kernel_sizes_and_strides(audio_length, depth,
                                                                                  kernel_count, kernel_size,
                                                                                  compression)

        if len(kernels) != depth:
            print(f"AudioConv_AE: only has depth={len(kernels)} instead of {depth}")
            return

        length = audio_length
        product = 1
        for i in range(len(kernels)):
            c = length / lengths[i]
            length = lengths[i]
            product *= strides[i]
            print(f"\tlayer {i+1}: kernel={kernels[i]:>3}, stride={strides[i]:>2}, length={lengths[i]:>6,}, compression={c:>6.1f}x, product={product:>6,}")

        self.expected_length = lengths[-1] + 1 # add 1 for the amplitude

        self.encoder = self.make_layers(False, kernel_count, kernels, strides)
        self.decoder = self.make_layers(True,  kernel_count, kernels, strides)

#        try:
        if True:
            self.encoded_shape, self.encoded_size = model_output_shape_and_size(self.encoder, [1, audio_length])
            print(f"\tencoded shape={self.encoded_shape}, size={self.encoded_size}")
            assert self.encoded_shape[0] == self.kernel_count
            assert self.encoded_shape[1] == self.expected_length, f"encoded shape[1]={self.encoded_shape[1]} instead of {self.expected_length}"
            assert self.encoded_size == self.expected_length * self.kernel_count, f"expected encoded_size={self.expected_length * self.kernel_count} but got {encoded_size}"

            decode_shape, decode_size = model_output_shape_and_size(self.decoder, self.encoded_shape)
            print(f"\tdecoded shape={decode_shape}, size={decode_size}")
        # except Exception as e:
        #     print(f"Model doesn't work: {e}")
        #     return
        #
        # except BaseException as e:
        #     print(f"Model is broken! {e}")
        #     return

        self.compression = audio_length / self.encoded_size
        print(f"AudioConv_AE {count_trainable_parameters(self):,} parameters, compression={self.compression:.1f}")

    def encode(self, x):
        batch_size = x.size(0)
        assert(x.size(1) == self.audio_length)
        x = x.view(batch_size, 1, self.audio_length)
        hiddens = self.encoder(x)
        periodically_display_2D_output(hiddens)
        return hiddens

    def decode(self, x):
        batch_size = x.size(0)
        decoded = self.decoder(x)
        decoded = decoded.view(batch_size, -1)
        len = decoded.size(1)
        missing = self.audio_length - len
        assert(missing >= 0)
        audio = F.pad(decoded, (0, missing))
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

        # torch.stft is not supported on MPS so we have to move things back to the CPU
        #now = time.time()
        inputs  = torch_stft(inputs)
        outputs = torch_stft(outputs)
        max_amp = inputs.abs().max()

        return basic_reconstruction_loss(inputs, outputs)/(max_amp**2)


if __name__ == "__main__":
    # audio_length, depth, kernel_count, kernel_size, compression
    print("\n\nTesting AudioConv_AE")
    model = AudioConv_AE(85_000, 3, 25, 168, 2.1)
    model.float()
    model.to(get_device())
    audio_length = 85_000
    print(f"encoded_shape={model.encoded_shape}")
    batched_input = torch.randn(7, audio_length).to(get_device())
    print(f"batched_input.shape={batched_input.shape}")
    loss, batched_output = model.forward_loss(batched_input)
    print(f"loss={loss}, batched_output={batched_output.shape}")
    assert batched_output.shape == batched_input.shape
    print("AudioConv_AE is OK!\n\n")

    print("\n\nTesting AudioConv_VAE")
    vae_sizes = [list(model.encoded_shape), 300, 200, 8]
    combined = CombinedVAE(model, vae_sizes)
    combined.float()
    combined.to(get_device())

    loss, batched_output = combined.forward_loss(batched_input)
    print(f"loss={loss}, batched_output={batched_output.shape}")
    assert batched_output.shape == batched_input.shape
    assert abs(loss - 100) < 1, f"loss={loss} is greater than expected"

    hidden_input = combined.auto_encoder.encode(batched_input)
    debug("hidden_input", hidden_input)

    loss, hidden_output = combined.vae.forward_loss(hidden_input)
    print(f"loss={loss}")
    debug("hidden_output", hidden_output)
    assert hidden_input.shape == hidden_output.shape


    print("AudioConv_VAE is OK!\n\n")

