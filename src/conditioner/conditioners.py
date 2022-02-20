import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from encdec import DecoderConvBlock
from utils.tf_utils import shape_list


class ConditionerNet(layers.Layer):
    """
    @residual_width: width of the down_sampling and residual stacks
    @residual_depth: change the receptive fields...
    @down_depth: list of down-sampling layers for each EncoderConvBlock
    @embed_width: most times just the latent_dim of the decoder input also
    """

    def __init__(self,
                 cond_shape,
                 bins,
                 embed_width,
                 residual_width,
                 residual_depth,
                 down_depth,
                 stride,
                 dilation_factor=1,
                 reverse_dilation=False,
                 dilation_cycle=None,
                 **kwargs):
        super(ConditionerNet, self).__init__(**kwargs)
        # assert depth == len(down_depth), f"Depth {depth} not Legit"
        # assert depth == len(strides), f"Depth {depth} not Legit"

        self.x_shape = cond_shape
        self.bins = bins  # Codebook size for latent codes
        self.depth = down_depth
        self.width = embed_width
        self.down_depth = down_depth
        self.stride = stride
        # width of the decoder
        self.embed_width = embed_width

        decoder_conv_block = lambda stride, down: DecoderConvBlock(embed_width, residual_width, residual_depth,
                                                                   stride=stride,
                                                                   dilation_factor=dilation_factor,
                                                                   reverse_dilation=reverse_dilation,
                                                                   down_depth=down, dilation_cycle=dilation_cycle)

        # # self.encoder_conv_blocks = [] #keras.Sequential()
        # self.model = keras.Sequential()
        #
        # # 1. Embedding map indices to d_model length embedding
        # self.cond_embedding = layers.Embedding(self.bins, self.width)
        #
        # # 2. Layer Normalization on Feature Dim
        # self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        #
        # # $$ reverse the shared down_sampling_layers input with encoder to be symmetric with the structure of encoder
        # # e.g. 128 -> 64 -> 32 - 32 -> 64 -> 128
        # # for layer, up_sampling_depth, stride in reversed(list(zip(list(range(self.depth)), down_depth, strides))):
        # #     print("Adding DecoderConvBlock: {}".format(layer))
        # #     # self.encoder_conv_blocks.append(encoder_conv_block(stride, down_sampling_depth))
        # #     self.model.add(decoder_conv_block(stride, up_sampling_depth))
        # self.model.add(decoder_conv_block(self.stride, self.depth))

        self.model = keras.Sequential([
            # 1. Embedding map indices to d_model length embedding
            layers.Embedding(self.bins, self.width),
            # 2. UpSampler Conv Net with Wavenet dilated Conv
            decoder_conv_block(self.stride, self.depth),
            # 3. Final layer normalization
            layers.LayerNormalization(axis=-1, epsilon=1e-6)
        ])

    def call(self, inputs, **kwargs):
        N, L = shape_list(inputs)
        tf.debugging.assert_equal(
            self.x_shape[0], L, message="Upper Level Shape Not match",
            summarize=None, name=None
        )

        # return self.model(inputs)
        out = self.model(inputs, )

        ## Assert Up-sampling
        _, L_out, _ = shape_list(out)
        tf.debugging.assert_equal(
            self.x_shape[0] * (self.stride ** self.down_depth), L_out, message="Upsampled Shape Not match",
            summarize=None, name=None
        )

        return out


if __name__ == '__main__':
    print('Conditioner module')

    # inputs = tf.random.normal([32, 128, 4])
    inputs = tf.random.normal([32, 128])

    conditioner = ConditionerNet(inputs.shape[1:], 100, 64, 32, 8, down_depth=3, stride=2, dilation_factor=3, dilation_cycle=4)

    outputs = conditioner(inputs)
    print(f"Out Shape: {outputs.shape}")
    conditioner.model.summary()

    # Print Detail Model Structure

    for cond_layer in conditioner.model.layers:
        print("-----{}-----".format(cond_layer.name))

        if 'conv_block' in cond_layer.name:
            # for l in cond_layer.model.layers[1::2]: # take only dilated layers
            for l in cond_layer.model.layers:
                print(f"------------{l.name}-------------")
                if 'dilated_resnet1d' in l.name:
                    for dilated_l in l.model.layers:
                        print(f"-----------------{dilated_l.name}----------------")
                        print([layer_.name for layer_ in dilated_l.model.layers])
            #     for layer in l.model.layers:
            #         print("---------{}---------".format(layer.name))
            #         for layer_ in layer.model.layers:
            #             print(layer_.name)
