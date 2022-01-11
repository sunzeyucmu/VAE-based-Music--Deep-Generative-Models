from itertools import chain
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from transformer.multi_head_attention import positional_encoding, create_look_ahead_mask

'''
Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.
'''


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


"""
    The Vanilla Transformer EncoderLayer
"""


class MHASelfAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(MHASelfAttentionBlock, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model,
                                             value_dim=d_model)  # key_dim == query_dim
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, mask, training=False):
        # self attention
        ## Masking: 1 indicate apply attention
        attn_output, attn_w = self.mha(x, x, x, attention_mask=mask, return_attention_scores=True, training=training)
        # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        '''
        Each of these sublayers has a residual connection around it followed by a layer normalization. 
        The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.
        '''
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_w


class MHABasedAutoregressiveModel(keras.Model):
    """
          @latent_dim
          @levels: number of stacked VQ-VAE (could be hierarchical like VQ-VAE-2, or independent like jukebox paper proposed)
        """

    def __init__(self,
                 target_vocab_size,  # num categorical distribution
                 width,  # d_model, model width
                 depth,  # num of MHA blocks
                 ffn_width=512,
                 heads=1,
                 maximum_pos_encoding=5000,
                 drop_out_rate=0.1,
                 context_length=None,  # Almost not needed for vanilla Attention
                 **kwargs):
        super(MHABasedAutoregressiveModel, self).__init__(**kwargs)

        self.context_length = context_length  # (T, )
        self.bins = target_vocab_size
        self.d_model = width
        self.num_layers = depth

        # 1. Embedding map indices to d_model length embedding
        self.x_embedding = layers.Embedding(self.bins, self.d_model)
        # 2. Positional Embedding to add timing info (TODO: is this applicable?)
        self.x_pos_encoding = positional_encoding(maximum_pos_encoding,
                                                  self.d_model)  # (1, maximum_pos_encoding, d_model)
        # 3. Multiple MHA self attention layers
        self.mha_attn_layers = [MHASelfAttentionBlock(self.d_model, heads, ffn_width, drop_out_rate)
                                # TODO: separate hyper for drop-out?
                                for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_out_rate)
        # 4. final layer to project to categorical distribution
        self.out = layers.Dense(self.bins)

    def call(self, x, training=False, x_cond=None):
        """

        :param x: (N, T) for audio latent code (T already compressed...)
        :param training:
        :param x_cond: up-sampled latent code from upper level
        :return:
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Self-Attention Look Ahead/Autoregressive mask; TODO: more mask types...
        autoregressive_mask = create_look_ahead_mask(seq_len, seq_len)

        x = self.x_embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.x_pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn_w = self.mha_attn_layers[i](x, autoregressive_mask, training)

            attention_weights[f'decoder_layer{i + 1}_attention'] = attn_w  # [N, H, T, T]
            # attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)

        # (N, T, D)
        final_output = self.out(x)

        return final_output, attention_weights


def loss_function(real, pred, loss_fn):
    """
  :param@real: target (B, L_max); padded 'index=0' will be masked out
  :param@pred: raw logits of network output (B, L, vocab_size)
  """
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_fn(real, pred)

    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask

    # return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    return tf.reduce_mean(loss_)  # loss per word

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    #
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    # accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    return tf.reduce_mean(accuracies)

# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
# def train_step(inputs):
#   tar_inp = tar[:, :-1]
#   tar_real = tar[:, 1:]
#
#   with tf.GradientTape() as tape:
#     predictions, _ = transformer([inp, tar_inp],
#                                  training = True)
#     loss = loss_function(tar_real, predictions)
#
#   gradients = tape.gradient(loss, transformer.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))



if __name__ == '__main__':
    print('Autoregressive module!')

    sample_in = tf.random.uniform((32, 4), dtype=tf.int64, minval=0, maxval=200)  # (N, T)

    automha = MHABasedAutoregressiveModel(context_length=sample_in.shape[1:], target_vocab_size=512, width=128, depth=2,
                                          heads=2)

    out, attn_w = automha(sample_in)

    print("Autoregressive Model Output: ", out.shape)
    print("Multi-head Self Attention: ", {k: v.shape for k, v in attn_w.items()})

    for _, v in attn_w.items():
        print(v.numpy()[0][-1])
