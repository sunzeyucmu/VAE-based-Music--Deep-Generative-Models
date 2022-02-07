import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from resnet import DilatedResnet1D

def print_dec_layer(decoder):
    for dec_conv in decoder.model.layers[:-1]:
        print("-----{}-----".format(dec_conv.name))
        for l in dec_conv.model.layers[1::2]: # take only dilated layers
            for layer in l.model.layers:
                print("---------{}---------".format(layer.name))
                for layer_ in layer.model.layers:
                    print(layer_.name)


class EncoderConvBlock(layers.Layer):
    """
    @embed_width: width of the down_sampling and residual stacks
    @down_depth: hop_length, num of down-sampling layer
    """

    def __init__(self, output_dim, embed_width, embed_depth, dilation_factor=1, stride=2, down_depth=4, **kwargs):
        super(EncoderConvBlock, self).__init__(**kwargs)

        self.model = keras.Sequential()

        self.kernel_size = stride * 2
        # For each EncoderResidualConv block
        # I: multiple down-sampling layers
        for i in range(down_depth):
            # 1. down-sampling
            self.model.add(layers.Conv1D(embed_width, self.kernel_size, strides=stride, padding="same"))
            # 2. dilated residual stack
            self.model.add(DilatedResnet1D(embed_width, embed_depth, dilation_factor=dilation_factor))

        # II: Proj Conv with kernel==3 to output channel size
        self.model.add(layers.Conv1D(output_dim, 3, strides=1, padding="same"))

    def call(self, inputs, **kwargs):
        return self.model(inputs)


class DecoderConvBlock(layers.Layer):
    """
    @embed_width: width of the down_sampling and residual stacks
    @down_depth: hop_length, num of down-sampling layer
    @reverse_dilation: normally true for decoder block
    """

    def __init__(self, output_dim, embed_width, embed_depth, dilation_factor=1, reverse_dilation=True, dilation_cycle=None, stride=2,
                 down_depth=4, **kwargs):
        super(DecoderConvBlock, self).__init__(**kwargs)

        self.model = keras.Sequential()
        self.kernel_size = stride * 2

        # For each DecoderResidualConv block
        # I. pre-process projection layer
        self.model.add(layers.Conv1D(embed_width, 3, strides=1, padding="same"))
        # II. Multiple up-sampling layers
        for i in range(down_depth):
            # 1.dilated residual stack
            self.model.add(DilatedResnet1D(embed_width, embed_depth, dilation_factor=dilation_factor, reverse_dilation=reverse_dilation, dilation_cycle=dilation_cycle))
            # 2. up-samling
            # - note remapping to output_dim for the last down-sampling layer
            self.model.add(layers.Conv1DTranspose(output_dim if i == (down_depth - 1) else embed_width, self.kernel_size
                                                  , strides=stride, padding="same"))

    def call(self, inputs, **kwargs):
        return self.model(inputs)


class Encoder(layers.Layer):
    """
    @residual_width: width of the down_sampling and residual stacks
    @residual_depth: change the receptive fields...
    @down_depth: list of down-sampling layers for each EncoderConvBlock
    """

    def __init__(self, output_dim, residual_width, residual_depth, depth, down_depth, strides, dilation_factor=1,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        assert depth == len(down_depth), f"Depth {depth} not Legit"
        assert depth == len(strides), f"Depth {depth} not Legit"

        self.depth = depth
        self.down_depth = down_depth
        self.strides = strides

        encoder_conv_block = lambda stride, down: EncoderConvBlock(output_dim, residual_width, residual_depth, stride=stride,
                                                                   dilation_factor=dilation_factor, down_depth=down)

        # self.encoder_conv_blocks = [] #keras.Sequential()
        self.model = keras.Sequential()
        for layer, down_sampling_depth, stride in zip(list(range(self.depth)), down_depth, strides):
            print("Adding EncoderConvBlock: {}".format(layer))
            # self.encoder_conv_blocks.append(encoder_conv_block(stride, down_sampling_depth))
            self.model.add(encoder_conv_block(stride, down_sampling_depth))

    def call(self, inputs, **kwargs):
        # TODO: validate shape after down-sampling
        # x = inputs

        # for encoder_conv_block in self.encoder_conv_blocks:
        #     x = encoder_conv_block(x)

        return self.model(inputs)


'''
    Decoder mirrors the same structure of encoder while doing up-sampling (Conv1D Transpose)
'''
class Decoder(layers.Layer):
    """
    @residual_width: width of the down_sampling and residual stacks
    @residual_depth: change the receptive fields...
    @down_depth: list of down-sampling layers for each EncoderConvBlock
    @embed_width: most times just the latent_dim of the decoder input also
    """

    def __init__(self, output_dim, embed_width, residual_width, residual_depth, depth, down_depth, strides
                 , dilation_factor=1, reverse_dilation=True, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        assert depth == len(down_depth), f"Depth {depth} not Legit"
        assert depth == len(strides), f"Depth {depth} not Legit"

        self.depth = depth
        self.down_depth = down_depth
        self.strides = strides
        # width of the decoder
        self.embed_width = embed_width

        decoder_conv_block = lambda stride, down: DecoderConvBlock(embed_width, residual_width, residual_depth, stride=stride,
                                                                   dilation_factor=dilation_factor, reverse_dilation=reverse_dilation,
                                                                   down_depth=down)

        # self.encoder_conv_blocks = [] #keras.Sequential()
        self.model = keras.Sequential()
        # $$ reverse the shared down_sampling_layers input with encoder to be symmetric with the structure of encoder
        # e.g. 128 -> 64 -> 32 - 32 -> 64 -> 128
        for layer, up_sampling_depth, stride in reversed(list(zip(list(range(self.depth)), down_depth, strides))):
            print("Adding DecoderConvBlock: {}".format(layer))
            # self.encoder_conv_blocks.append(encoder_conv_block(stride, down_sampling_depth))
            self.model.add(decoder_conv_block(stride, up_sampling_depth))

        # The final Conv1D projects to the desired number of audio channels
        self.model.add(layers.Conv1D(output_dim, 3, strides=1, padding='same'))

    def call(self, inputs, **kwargs):
        return self.model(inputs)


if __name__ == '__main__':
    print('Encoder Conv Block module')

    model = EncoderConvBlock(64, 64, 4, dilation_factor=3, down_depth=5)
    inputs = tf.random.normal([32, 1024, 4])
    outpus = model(inputs)
    print(outpus.shape)

    print(model.model.summary())

    # encoder = Encoder(64, 64, 4, 1, down_depth=[5], strides=[2], dilation_factor=3)
    # Multiple Encoder Conv layers
    encoder = Encoder(64, 64, 4, 2, down_depth=[5, 3], strides=[2, 2], dilation_factor=3)

    enc_output = encoder(inputs)

    encoder.model.summary()

    # decoder = Decoder(1, 64, 64, 4, depth=1, down_depth=[5], strides=[2], dilation_factor=3)
    decoder = Decoder(1, 64, 64, 4, depth=2, down_depth=[5, 3], strides=[2, 2], dilation_factor=3)

    dec_output = decoder(enc_output)

    decoder.model.summary()

     # Debug dilation stack
    for dec_conv in decoder.model.layers[:-1]:
        print("-----{}-----".format(dec_conv.name))
        for l in dec_conv.model.layers[1::2]: # take only dilated layers
            for layer in l.model.layers:
                print("---------{}---------".format(layer.name))
                for layer_ in layer.model.layers:
                    print(layer_.name)