import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class ResnetConv1DBlock(layers.Layer):
    def __init__(self, input_dim, filters, dilation=1, **kwargs):
        super(ResnetConv1DBlock, self).__init__(**kwargs)

        self.model = keras.Sequential([
            layers.ReLU(),
            layers.Conv1D(filters, 3, dilation_rate=dilation, padding="same",
                          name="dilated_cov1d_dr-{}".format(dilation)),
            layers.ReLU(),
            layers.Conv1D(input_dim, 3, dilation_rate=1, padding="same")
        ])

        # self.stack = [layers.ReLU(),
        #     layers.Conv1D(filters, 3, dilation_rate=dilation, padding="same", name="dilated_cov1d_dr-{}".format(dilation)),
        #     layers.ReLU(),
        #     layers.Conv1D(input_dim, 3, dilation_rate=1, padding="same")
        # ]

    def call(self, input_tensor, **kwargs):
        # residual connection
        # return input_tensor + self.model(input_tensor)
        return layers.add([input_tensor, self.model(input_tensor)])

        # stacked version
        # x = input_tensor
		#
        # for layer in self.stack:
        #     x = layer(x)
		#
        # return layers.add([input_tensor, x])


class DilatedResnet1D(layers.Layer):
    def __init__(self, input_dim, depth, dilation_factor=1, reverse_dilation=False, **kwargs):
        super(DilatedResnet1D, self).__init__(**kwargs)
        # stack of dilated residual blocks
        blocks = [ResnetConv1DBlock(input_dim, input_dim, dilation=dilation_factor ** d) for d in range(depth)]

        # for decoder stack... dilations constracts by a factor of 3 down to 1 at the last block
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = keras.Sequential(blocks)  # no need implicit unpacking

    def call(self, input, **kwargs):
        return self.model(input)


if __name__ == '__main__':
    print('Residual Block module')

    test_res_block = ResnetConv1DBlock(64, 32, dilation=1)

    test_input = tf.random.uniform([32, 200, 64])
    test_output = test_res_block(test_input)

    print(test_output.shape)

    # test_res_block.summary()
    dilated_res_stack = DilatedResnet1D(32, 3, dilation_factor=3, reverse_dilation=True)
    inputs = keras.Input(shape=(200, 32))  # TC
    # outputs = DilatedResnet1D(32, 3, dilation_factor=3, reverse_dilation=False)(inputs)
    outputs = dilated_res_stack(inputs)

    model = keras.Model(inputs, outputs)

    model.summary()

    dilated_res_stack.model.summary()

    # Debug dilation stack
    for layer in dilated_res_stack.model.layers:
        print("---------{}---------".format(layer.name))
        for layer_ in layer.model.layers:
            print(layer_.name)
