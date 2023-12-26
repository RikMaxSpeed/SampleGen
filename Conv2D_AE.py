import torch
import torch.nn as nn
import torch.nn.functional as F
from dask import layers

from ModelUtils import rnn_size, periodically_display_2D_output
from VariationalAutoEncoder import reconstruction_loss, VariationalAutoEncoder
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters, model_output_shape_and_size, device
from Debug import *
from MakeSTFTs import freq_buckets, sequence_length
import copy

def convolution_output_size(input, kernel, stride, pad):
    return (input + 2*pad - kernel) // stride + 1

def transposed_convolution_output_size(input, kernel, stride, pad):
    return stride * (input - 1) + kernel - 2 * pad

def max_stride(kernel_size, dimension, min_dim):
    if dimension < min_dim:
        return 1

    return max(2, kernel_size // 3)

def max_kernel(dimension, kernel_size):
    return min(kernel_size, max(dimension // 4, 1))

def pad_size(dimension, kernel, stride):
    assert stride <= kernel
    return 0 # for some reason my padding broke the MPS back-end, whilst it works fine on CPU :(

    for pad in range(kernel):
        size = convolution_output_size(dimension, kernel, stride, pad)
        transposed = transposed_convolution_output_size(size, kernel, stride, pad)
        if transposed == size:
            #print(f"padding={pad} for dimension={dimension}, kernel={kernel}, stride={stride}")
            return pad

    #print(f"padding failed for dimension={dimension}, kernel={kernel}, stride={stride}")
    return 0

class Conv2DAutoEncoder(nn.Module):
    @staticmethod
    def count_conv2d_parameters(layers):
        sum = 0
        #print("count_conv2d_parameters:")
        for layer in layers:
            kernel_size = layer['kernel_size']
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            parameters = kernel_size[0] * kernel_size[1] * in_channels * out_channels + out_channels
            sum += parameters
            #print(f"\tlayer: {layer}, parameters={parameters:,}")

        #print(f"total={sum:,}")
        return sum

    @staticmethod
    def infer_conv2d_layers(layer_count, kernel_count, kernel_size, transpose):
        layers = []
        k_size = kernel_size

        # Use the input size to determine plausible kernels
        freqs = freq_buckets
        steps = sequence_length

        for i in range(layer_count):

            inputs = kernel_count
            outputs = kernel_count

            if i == 0:
                if transpose:
                    outputs = 1
                else:
                    inputs = 1

            k_size = (max_kernel(freqs, kernel_size), max_kernel(steps, kernel_size))

            if i == 0:
                stride = (1, 1) # ensures the last decoded layer is smooth
            else:
                stride = (max_stride(k_size[0], freqs, freq_buckets // 16), max_stride(k_size[1], steps, sequence_length // 8))

            if k_size == (1, 1):
                break

            #pad = 'same' # unfortunately not supported for strided convolutions...
            pad = (pad_size(freqs, k_size[0], stride[0]), pad_size(steps, k_size[1], stride[1]))

            layer = {'in_channels': inputs,
                     'out_channels': outputs,
                     'kernel_size': k_size,
                     'stride': stride,
                     'padding': pad}

            layers.append(layer)

            freqs = convolution_output_size(freqs, k_size[0], stride[0], pad[0])
            steps = convolution_output_size(steps, k_size[1], stride[1], pad[1])
            kernel_size = max(kernel_size // 2, 2)

        #print(f"expected output: {freqs} x {steps}") # this is correct :)

        if transpose:
            layers = list(reversed(layers))

        return layers

    @staticmethod
    def display(name, layer_count, kernel_count, kernel_size, layers):
        print(f"{name}: layer_count={layer_count}, kernel_count={kernel_count}, kernel_size={kernel_size}")
        for layer in layers:
            print(f"\t{layer}")

    @staticmethod
    def infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size):
        encode_layers = Conv2DAutoEncoder.infer_conv2d_layers(layer_count, kernel_count, kernel_size, False)
        decode_layers = Conv2DAutoEncoder.infer_conv2d_layers(layer_count, kernel_count, kernel_size, True)

        return encode_layers, decode_layers

    @staticmethod
    def approx_trainable_parameters(layer_count, kernel_count, kernel_size):
        encode_layers, decode_layers = Conv2DAutoEncoder.infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size)

        if False:
            Conv2DAutoEncoder.display("encoder", layer_count, kernel_count, kernel_size, encode_layers)
            Conv2DAutoEncoder.display("decoder", layer_count, kernel_count, kernel_size, decode_layers)

        encoder_size = Conv2DAutoEncoder.count_conv2d_parameters(encode_layers)
        decoder_size = Conv2DAutoEncoder.count_conv2d_parameters(decode_layers)
        return encoder_size + decoder_size

    @staticmethod
    def build_conv2d_network(layers, transpose):
        function = nn.ConvTranspose2d if transpose else nn.Conv2d
        sequence = []
        for layer in layers:
            sequence.append(function(in_channels=layer['in_channels'],
                                     out_channels=layer['out_channels'],
                                     kernel_size=layer['kernel_size'],
                                     stride=layer['stride'],
                                     padding=layer['padding']))

        # if transpose:
        #     sequence.append(nn.Sigmoid()) # huge impact on accuracy :(

        return nn.Sequential(*sequence)

    def __init__(self, freq_buckets, sequence_length, layer_count, kernel_count, kernel_size):
        super(Conv2DAutoEncoder, self).__init__()

        self.freq_buckets = freq_buckets
        self.sequence_length = sequence_length

        encode_layers, decode_layers = Conv2DAutoEncoder.infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size)

        self.encoder = Conv2DAutoEncoder.build_conv2d_network(encode_layers, False)
        self.decoder = Conv2DAutoEncoder.build_conv2d_network(decode_layers, True)

        try:
            input_shape = (freq_buckets, sequence_length)
            self.encode_shape, self.encoded_size = model_output_shape_and_size(self.encoder, input_shape)
            if kernel_count == 1:
                self.encode_shape = (1,) + self.encode_shape

            assert len(self.encode_shape) == 3
            print("encode_shape=", self.encode_shape)

            self.decode_shape, self.decode_size = model_output_shape_and_size(self.decoder, self.encode_shape)
            #print(f"decode.shape={self.decode_shape}")
            #assert self.decode_shape[1] == freq_buckets, f"decoded {self.decode_shape[1]} frequencies  instead of {freq_buckets}"

        except BaseException as e:
            print(f"Model doesn't work: {e}")
            self.encode_shape = ()
            self.encoded_size = 0
            self.compression = 0
            return

        input_size = freq_buckets * sequence_length
        self.compression = input_size/self.encoded_size
        print(f"Conv2DAutoEncoder: input={input_size:,}, encoded={self.encoded_size:,}, compression={self.compression:.1f}")

    def encode(self, x):
        x = x.unsqueeze(1)
        #debug("encode.x", x)
        encoded = self.encoder(x)
        #debug("encoded", encoded)
        return encoded.flatten(start_dim=1)

    def decode(self, x):
        x = x.view(x.size(0), *self.encode_shape)

        #debug("decode.x", x)
        decoded = self.decoder(x).squeeze(dim=1)
        #debug("decoded", decoded)

        # Pad the output with zeros to reach the desired size
        missing = self.sequence_length - decoded.size(2)
        if missing > 0:
            #print(f"adding {missing} time-steps")
            decoded = F.pad(decoded, (0, missing))

        missing = self.freq_buckets - decoded.size(1)
        if missing > 0:
            #print(f"adding {missing} missing frequencies")
            decoded =F.pad(decoded, (0, 0, 0, missing))

        assert decoded.size(1) == self.freq_buckets
        assert decoded.size(2) == self.sequence_length

        return decoded

    def forward(self, inputs):
        latent = self.encode(inputs)
        outputs = self.decode(latent)
        return outputs

    def forward_loss(self, inputs):
        outputs = self.forward(inputs)
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs




if __name__ == '__main__':
    layer_count = 1
    kernel_count = 5
    kernel_size = 4

    approx_params = Conv2DAutoEncoder.approx_trainable_parameters(layer_count, kernel_count, kernel_size)
    #print(f"approx_params: {approx_params}")

    freq_buckets= 1024
    sequence_length = 80
    model = Conv2DAutoEncoder(freq_buckets, sequence_length, layer_count, kernel_count, kernel_size)
    model.float() # ensure we're using float32 and not float64
    model.to(device)

    #print(model)
    exact_params = count_trainable_parameters(model)
    #print(f"exact_params: {exact_params}")
    assert(approx_params == exact_params)

    batch_size = 7
    input = torch.randn((batch_size, freq_buckets, sequence_length))
    input = input.to(device)
    #debug("input", input)

    encoded = model.encode(input)
    #debug("encoded", encoded)

    output = model.decode(encoded)
    #debug("output", output)

    assert input.shape == output.shape
