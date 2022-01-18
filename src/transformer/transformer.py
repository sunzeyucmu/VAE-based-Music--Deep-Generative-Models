import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.transformer.factorized_attention import FactorizedAttention

'''
    Each layer of Transformer is a residual-attention block
'''


class ResidualAttnBlock(layers.Layer):
    def __init__(self,
                 ctc_len,
                 num_heads,
                 d_model,
                 blocks,
                 attn_func=0,
                 m_attn=0.25,
                 m_mlp=1.0,
                 rate=0.1,
                 **kwargs):
        super(ResidualAttnBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.attn_func = attn_func  # index

        self.fmha = FactorizedAttention(ctc_len, num_heads, d_model, blocks, attn_func=attn_func, m_attn=m_attn,
                                        drop_out_rate=rate)

        self.mlp = layers.Dense(tf.cast(self.d_model * m_mlp, tf.int64))

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, return_attention_weights=False, **kwargs):
        """

        :param return_attention_weights:
        :param inputs: [N, T, D]
        :param training:
        :param kwargs:
        :return:
        """
        # LayerNorm + Attention
        if return_attention_weights:
            res1, attn_weights = self.fmha(self.layernorm1(inputs), training=training,
                                           return_attention_weights=return_attention_weights,
                                           **kwargs)
        else:
            res1 = self.fmha(self.layernorm1(inputs), training=training,
                             return_attention_weights=return_attention_weights,
                             **kwargs)
        # LayerNorm + MLP
        res2 = self.mlp(self.layernorm2(inputs + res1), training=training)

        out = res2 + res1 + inputs

        if return_attention_weights:
            return out, attn_weights
        return out


class FactorizedTransformer(layers.Layer):
    def __init__(self,
                 ctc_len,
                 num_heads,
                 depth,
                 d_model,
                 blocks,
                 attn_stacks,  # index of attention stack(s) structure
                 m_attn=0.25,
                 m_mlp=1.0,
                 rate=0.1,
                 **kwargs):
        super(FactorizedTransformer, self).__init__(**kwargs)
        self.d_model = d_model
        self.depth = depth

        # Stacks of attn_func
        ## decided by hyper-param:attn_stacks of prior/upsampler model
        ## only one row gets selected
        self.attn_func = {
            0: lambda d: [0, 1][d % 2],  # Alternate row and column attn
            1: lambda d: [0, 1, 2][d % 3]  # Alternate row, column and previous row attn
            # TODO: more combination
        }[attn_stacks]

        res_attn_block_func = lambda l: ResidualAttnBlock(ctc_len=ctc_len, num_heads=num_heads, d_model=d_model,
                                                          blocks=blocks, attn_func=self.attn_func(l),
                                                          m_attn=m_attn, m_mlp=m_mlp, rate=rate)

        # TODO: structure printing func...
        self.model = keras.Sequential()
        self.model.add(layers.Input(shape=(ctc_len, self.d_model)))

        for i in range(depth):
            self.model.add(res_attn_block_func(i))

    def call(self, inputs, **kwargs):
        # assert tf.shape(inputs)[-1] == self.d_model
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1], self.d_model, message="Input Width not consistent with model Width", summarize=None, name=None
        )
        # return self.model(inputs, **kwargs)

        attention_weights = {}
        x = inputs
        for i, layer in enumerate(self.model.layers):
            if i + 1 == self.depth or i + 1 == 1:
                x, att_w = layer(x, return_attention_weights=True, **kwargs)
                attention_weights['transformer_layer_{}_attention'.format(i)] = att_w # (N, H, L, L)
            else:
                x = layer(x, return_attention_weights=False, **kwargs)

        return x, attention_weights


if __name__ == '__main__':
    print("Transformer Module!")

    transformer = FactorizedTransformer(ctc_len=16, num_heads=4, d_model=48, blocks=4, attn_stacks=1, depth=6)

    for layer in transformer.model.layers:
        print(layer.attn_func, layer.fmha.attn_type)

    inputs = tf.random.normal([4, 16, 48])

    outputs, attn_w = transformer(inputs, training=True)

    print(outputs.shape)
    for k, v in attn_w.items():
        print(k)
        print(v.numpy()[0][0])

    sample_outputs, _ = transformer(tf.random.normal([4, 3, 48])) # TODO: During Sample, current sample_len < block_len (16/4==4 in this case)

    print(sample_outputs.shape)
