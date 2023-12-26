import torch
import torch.nn as nn
import torch.nn.functional as F
from Debug import debug
from ModelUtils import conv1d_size, periodically_display_2D_output, model_output_shape_and_size
from VariationalAutoEncoder import reconstruction_loss, VariationalAutoEncoder
from ModelUtils import interpolate_layer_sizes, count_trainable_parameters


class AudioConv_AE(nn.Module):  # no VAE

    @staticmethod
    def approx_trainable_parameters(depth, kernel_count, outer_kernel_size, inner_kernel_size):
        encode = conv1d_size(1, kernel_count, outer_kernel_size) \
                  + (depth - 1) * conv1d_size(kernel_count, kernel_count, inner_kernel_size)

        return 2 * encode

    def make_layers(self, is_decoder, depth, kernel_count, outer_kernel_size, inner_kernel_size):
        assert(depth >= 1)
        layers = []
        # Outer layer
        stride = outer_kernel_size // 2
        if is_decoder:
            layers.append(torch.nn.ConvTranspose1d(kernel_count, 1, outer_kernel_size, stride=stride))
        else:
            layers.append(torch.nn.Conv1d(1, kernel_count, outer_kernel_size, stride=stride))

        stride = 2
        for i in range(1, depth):
            if is_decoder:
                layers.append(torch.nn.ConvTranspose1d(kernel_count, kernel_count, inner_kernel_size, stride=stride))
            else:
                layers.append(torch.nn.Conv1d(kernel_count, kernel_count, inner_kernel_size, stride=stride))

        if is_decoder:
            layers.reverse()

        return nn.Sequential(*layers)

    def __init__(self, audio_length, depth, kernel_count, outer_kernel_size, inner_kernel_size):
        super(AudioConv_AE, self).__init__()

        self.audio_length = audio_length
        self.kernel_count = kernel_count

        self.encoder = self.make_layers(False, depth, kernel_count, outer_kernel_size, inner_kernel_size)
        self.decoder = self.make_layers(True,  depth, kernel_count, outer_kernel_size, inner_kernel_size)

        try:
            encoded_shape, encoded_size = model_output_shape_and_size(self.encoder, [1, audio_length])
            print(f"encoded shape={encoded_shape}, size={encoded_size}")

            decode_shape, decode_size = model_output_shape_and_size(self.decoder, encoded_shape)
            print(f"decoded shape={decode_shape}, size={decode_size}")
        except BaseException as e:
            print(f"Model doesn't work: {e}")
            self.compression = 0
            return


        #self.compression = audio_length / outer_kernel_size # approx
        self.compression = audio_length / encoded_size
        print(f"AudioConv_AE {count_trainable_parameters(self):,} parameters, compression={self.compression:.1f}")

        print(self)

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
        loss = reconstruction_loss(inputs, outputs)
        return loss, outputs
