import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dummy_module():
    return "HHH"


'''
Self-Attention Mask
'''


def create_look_ahead_mask(q_len, k_len):
    # mask = 1 - tf.linalg.band_part(tf.ones((q_len, k_len)), -1, 0) #keep the Lower triangular part.
    mask = tf.linalg.band_part(tf.ones((q_len, k_len)), -1, 0)  # 0 the upper triangular part.
    return mask  # (seq_len, seq_len)


def get_angles(pos, i, d_model):
    # 1/1000^{2i*/d_model}
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    # pos/1000^{2i*/d_model
    # print(pos)
    # print(angle_rates)
    return pos * angle_rates


'''
    @return (1, position, d_model)
'''


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        # Latent Embedding Space: K x D/L
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.num_embeddings, self.embedding_dim), dtype="float32"
            ),
            trainable=True,
            name="positional_embedding",
        )  # The Model Weights

    def call(self, inputs, **kwargs):
        seq_len = tf.shape(inputs)[1]
        return tf.expand_dims(self.embeddings[:seq_len, :], axis=0)

    def get_embeddings(self, sequence_length, **kwargs):
        pos_embeddings = tf.expand_dims(self.embeddings[:sequence_length, :], axis=0)
        return pos_embeddings


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # tf.print(f'[DEBUG] LR Schedule: Step: {step}')
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        # lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        #
        # # if tf.equal(tf.rank(step), tf.constant(0)):
        # tf.summary.scalar('lr', data=lr, step=tf.cast(step, tf.int64))

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    print("Multi-Head Attention Module!")

    # Specified Axes
    mha = layers.MultiHeadAttention(num_heads=2, key_dim=4, attention_axes=(2, 3))
    input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])  # (T=3, dim=4)
    output_tensor, attn_w = mha(input_tensor, input_tensor, return_attention_scores=True)
    mha = layers.MultiHeadAttention(num_heads=1, key_dim=4, value_dim=4)
    query = tf.keras.Input(shape=[8, 12])
    key = tf.keras.Input(shape=[4, 10])
    output, attn_w = mha(query, key, return_attention_scores=True)
    print(output_tensor.shape)
    print(attn_w.shape)
