import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from encdec import DecoderConvBlock
from utils.tf_utils import shape_list


class LabelConditioner(layers.Layer):

    def __init__(self,
                 genre_bins,
                 width,
                 **kwargs):
        super(LabelConditioner, self).__init__(**kwargs)

        self.model = width
        self.genre_bins = genre_bins

        self.genre_emb = layers.Embedding(self.genre_bins, self.model, input_length=1)

    def call(self, y, **kwargs):
        """

        :param y: (N, ) - batch of labels, integer indice for each
        :param kwargs:
        :return:
        """
        # tf.debugging.assert_equal(
        #     len(tf.shape(y)), 2,
        #     message=f"Expect label",
        #     summarize=None, name=None
        # )

        # TODO: Multiple Genres....
        out = self.genre_emb(y, **kwargs)

        # out = tf.expand_dims(out, axis=1)
        out = out[:, tf.newaxis, :]
        tf.debugging.assert_equal(
            shape_list(out), [shape_list(y)[0], 1, self.model], message=f"Genre Embedding Shape Not matching expectation: {shape_list(out)}",
            summarize=None, name=None
        )

        return out





if __name__ == '__main__':
    print('Label Conditioner module')

    label_conditioner = LabelConditioner(10, 32)
    y_in = tf.random.uniform((4,), dtype=tf.int64, minval=0, maxval=9)
    y_out = label_conditioner(y_in, training=False)

    print(y_out)
