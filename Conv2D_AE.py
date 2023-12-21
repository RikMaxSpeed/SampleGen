import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils import rnn_size, periodically_display_2D_output
from VariationalAutoEncoder import reconstruction_loss, VariationalAutoEncoder
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters, model_output_shape_and_size, device
from Debug import *
import copy


class Conv2DAutoEncoder(nn.Module):
    @staticmethod
    def count_conv2d_parameters(layers):
        sum = 0
        #print("count_conv2d_parameters:")
        for layer in layers:
            kernel_size = layer['kernel_size']
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            parameters = (kernel_size ** 2) * in_channels * out_channels + out_channels
            sum += parameters
            #print(f"\tlayer: {layer}, parameters={parameters:,}")

        #print(f"total={sum:,}")
        return sum

    @staticmethod
    def infer_conv2d_layers(layer_count, kernel_count, kernel_size, transpose):
        layers = []
        for i in range(layer_count):
            k_size = max(kernel_size - i, 2)

            inputs = kernel_count
            outputs = kernel_count

            if i == 0:
                if transpose:
                    outputs = 1
                else:
                    inputs = 1

            overlap = 1 #k_size // 3 # make this a hyper-parameter too?
            stride = max(k_size - overlap, 2)
            #pad = 'same' # unfortunately not supported for strided convolutions...
            pad = k_size - 1 # approximate, gets us close enough
            layer = {'in_channels': inputs,
                     'out_channels': outputs,
                     'kernel_size': k_size,
                     'stride': stride,
                     'padding': pad}

            layers.append(layer)

        if transpose:
            layers = list(reversed(layers))

        #print(f"layer_count={layer_count}, kernel_count={kernel_count}, kernel_size={kernel_size}")
        #print(f"layers: {layers}")

        return layers

    @staticmethod
    def infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size):
        encode_layers = Conv2DAutoEncoder.infer_conv2d_layers(layer_count, kernel_count, kernel_size, False)
        decode_layers = Conv2DAutoEncoder.infer_conv2d_layers(layer_count, kernel_count, kernel_size, True)

        return encode_layers, decode_layers

    @staticmethod
    def approx_trainable_parameters(layer_count, kernel_count, kernel_size):
        encode_layers, decode_layers = Conv2DAutoEncoder.infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size)
        encoder_size = Conv2DAutoEncoder.count_conv2d_parameters(encode_layers)
        decoder_size = Conv2DAutoEncoder.count_conv2d_parameters(decode_layers)
        return encoder_size + decoder_size

    @staticmethod
    def build_conv2d_network(layers, transpose):
        sequence = []
        function = nn.ConvTranspose2d if transpose else nn.Conv2d
        for layer in layers:
            sequence.append(function(in_channels=layer['in_channels'],
                                               out_channels=layer['out_channels'],
                                               kernel_size=layer['kernel_size'],
                                               stride=layer['stride'],
                                               padding=layer['padding']))

        sequence.append(nn.Sigmoid())
        return nn.Sequential(*sequence)

    def __init__(self, freq_buckets, sequence_length, layer_count, kernel_count, kernel_size):
        super(Conv2DAutoEncoder, self).__init__()

        self.freq_buckets = freq_buckets
        self.sequence_length = sequence_length

        encode_layers, decode_layers = Conv2DAutoEncoder.infer_encode_and_decode_layers(layer_count, kernel_count, kernel_size)

        self.encoder = Conv2DAutoEncoder.build_conv2d_network(encode_layers, False)
        self.decoder = Conv2DAutoEncoder.build_conv2d_network(decode_layers, True)

        self.encode_shape, self.encoded_size = model_output_shape_and_size(self.encoder, (freq_buckets, sequence_length))
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

        assert decoded.size(1) == self.freq_buckets # else we're in trouble...

        missing = self.sequence_length - decoded.size(2)
        if missing > 0:
            #print(f"missing={missing}")
            decoded = F.pad(decoded, (0, missing))
            #debug("decoded.padded", decoded)

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
