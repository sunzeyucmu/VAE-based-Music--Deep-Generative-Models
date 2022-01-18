import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.transformer.multi_head_attention import create_look_ahead_mask
from utils.tf_utils import shape_list


class FactorizedAttention(layers.Layer):
    def __init__(self,
                 ctc_len,
                 num_heads,
                 d_model,
                 blocks,
                 attn_func=0,
                 m_attn=0.25,
                 drop_out_rate=0.0,
                 **kwargs):
        super(FactorizedAttention, self).__init__(**kwargs)
        # Block level model width
        self.width = tf.cast(d_model * m_attn, tf.int64)
        assert self.width % num_heads == 0
        self.num_heads = num_heads
        self.ctx_len = ctc_len
        self.blocks = blocks
        # self.attn_type = attn_func # Factorized Attention Type

        assert self.ctx_len % blocks == 0
        self.block_len = self.ctx_len // blocks  # $ l: strides_len

        # TODO: separate query, key value
        self.qkv_conv = layers.Conv1D(self.width * 3, 3, strides=1, dilation_rate=1, padding="same")
        # Note that 'key_dim/query_dim and value_dim' being size of each $attention head$ for query and key.
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.width // num_heads,
                                             value_dim=self.width // num_heads)  # key_dim == query_dim
        # Select Factorized Attention type
        self.attn, self.attn_type = {
            0: (self.row_attn, "row"),
            1: (self.col_attn, "col"),
            2: (self.prev_row_attn, "prev row")
            # TODO: more attention type; e.g. dec-enc attention
        }[attn_func]

        # Projection back to input_width
        self.proj = layers.Dense(d_model)
        self.attn_dropout = layers.Dropout(drop_out_rate)

    def call(self, inputs, training=False, return_attention_weights=False, **kwargs):
        """

        :param return_attention_weights:
        :param inputs: (N, T, D)
        :param training:
        :param kwargs:
        :return:
        """
        ctx_len = tf.shape(inputs)[1]

        x = self.qkv_conv(inputs, training=training) # (N, T, 3*d)
        query, key, value = tf.split(x, num_or_size_splits=3, axis=-1) # (N, T, d)

        attn_out, attn_w = self.attn(query, key, value)
        out = self.proj(attn_out) # (N, T, D)

        if return_attention_weights:
            return self.attn_dropout(out, training=training), attn_w
        return self.attn_dropout(out, training=training)


    def row_attn(self, q, k, v, sample=False):
        """

        :param q:
        :param k:
        :param v:
        :param sample: TODO
        :return:
        """

        N, L, D = shape_list(k)
        Lq = shape_list(q)[1]
        # TODO
        assert Lq == L
        num_blocks = Lq // self.block_len

        # reshape the input sequence (N, L, D) into 2D sequence [blocks, block length]
        q = tf.reshape(q, [N * num_blocks, self.block_len, D])
        # TODO: if Lq != L, then the batch dim would be different!
        k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        v = tf.reshape(v, [N * num_blocks, self.block_len, D])

        attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
                                     return_attention_scores=True)  # , training=training)

        return tf.reshape(mha_out, [N, Lq, D]), mha_attn


    def col_attn(self, q, k, v, sample=False):
        """

        :param q:
        :param k:
        :param v:
        :param sample: TODO
        :return:
        """

        N, L, D = shape_list(k)
        Lq = shape_list(q)[1]
        # TODO
        assert Lq == L
        num_q_blocks = Lq // self.block_len
        num_kv_blocks = L // self.block_len # q, kv num blocks no need to be equal
        # print("Before Col Transpose to Row: ", tf.reshape(q, [N, num_q_blocks, self.block_len, D])[0,...,0])
        q = tf.transpose(tf.reshape(q, [N, num_q_blocks, self.block_len, D]), [0, 2, 1, 3]) # (N, l, blocks, D)
        # print("After Col Transpose to Row: ", q[0, ..., 0])
        k = tf.transpose(tf.reshape(k, [N, num_kv_blocks, self.block_len, D]), [0, 2, 1, 3]) # (N, l, blocks, D)
        k = tf.transpose(tf.reshape(v, [N, num_kv_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)

        q = tf.reshape(q, [N*self.block_len, num_q_blocks, D]) # (N*l, blocks, D), l*blocks = Tq
        k = tf.reshape(k, [N*self.block_len, num_kv_blocks, D])
        v = tf.reshape(v, [N*self.block_len, num_kv_blocks, D])

        attn_mask = create_look_ahead_mask(num_q_blocks, num_kv_blocks)

        mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
                                     return_attention_scores=True)  # , training=training)

        return tf.reshape(mha_out, [N, Lq, D]), mha_attn

    def prev_row_attn(self, q, k, v, sample=False):
        """

        :param q:
        :param k:
        :param v:
        :param sample: TODO
        :return:
        """

        N, L, D = shape_list(k)
        Lq = shape_list(q)[1]
        # TODO
        assert Lq == L
        num_blocks = Lq // self.block_len

        q = tf.reshape(q, [N * num_blocks, self.block_len, D])

        # 1.remove current block from bottom
        # print("Before Shift Blocks: ", tf.reshape(k, [N, num_blocks, self.block_len, D])[0, ..., 0].numpy())
        k = tf.reshape(k, [N, num_blocks, self.block_len, D])[:, :-1, :, :]
        # print("After Shift Blocks: ", k[0, ..., 0].numpy())
        v = tf.reshape(v, [N, num_blocks, self.block_len, D])[:, :-1, :, :]
        # 2. pad dim:-3, the num_blocks dimension with extra block at top
        # print("Before Pad: ", k[0, ..., 0].numpy())
        k = tf.pad(k, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # print("After Pad: ", k[0, ..., 0].numpy())
        v = tf.pad(v, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # 1 && 2 combined ==> shift forward k, v blocks by one block
        k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        v = tf.reshape(v, [N * num_blocks, self.block_len, D])
        # TODO: if Lq < L?

        attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
                                     return_attention_scores=True)  # , training=training)

        return tf.reshape(mha_out, [N, Lq, D]), mha_attn


if __name__ == '__main__':
    print("Factorized Attention Module!")

    np.set_printoptions(precision=2)

    # Sequence of length 16
    query = tf.random.normal([4, 16, 48])
    key = tf.random.normal([4, 16, 48])
    value = tf.random.normal([4, 16, 48])

    # attn_mask = create_look_ahead_mask(query.shape[1], key.shape[1])
    #
    # print("Attention Mask(Auto-Regressive): ", attn_mask)
    #
    # mha = layers.MultiHeadAttention(num_heads=4, key_dim=4, value_dim=4)  # key_dim == query_dim
    #
    # query_input = tf.keras.Input(shape=[8, 12])
    # value_input = tf.keras.Input(shape=[4, 12])
    # output, attn_w = mha(query_input, value_input, return_attention_scores=True)
    # # B, T, E (E being the output dimension of the query if output_shape not specified)
    # print(output.shape)  # (B, Tq, E(query_dim))
    # print(attn_w.shape)  # (B, H, Tq, Tkv)
    #
    # mha_out, mha_attn = mha(query=query, key=key,
    #                         value=value, return_attention_scores=True, attention_mask=attn_mask)
    #
    # print(mha_attn[0][0].numpy())  # (pick first attention Head
    # # print(mha_attn[1][0].numpy()) # Validate batch dim...
    #
    # ## Block Attention: Reshape sequence into attention block (b x l)
    # L = tf.shape(query)[1]
    # b = 4
    # l = L // b
    # query_block = tf.reshape(query,
    #                          [tf.shape(query)[0] * b, l, tf.shape(query)[-1]])  # extend blocks to batch dimension
    # key_block = tf.reshape(key, [tf.shape(query)[0] * b, l, tf.shape(query)[-1]])
    # value_block = tf.reshape(value, [tf.shape(query)[0] * b, l, tf.shape(query)[-1]])
    #
    # attn_mask_block = create_look_ahead_mask(query_block.shape[1], key_block.shape[1])
    #
    # mha_out_block, mha_attn_block = mha(query=query_block, key=key_block,
    #                                     value=value_block, return_attention_scores=True, attention_mask=attn_mask_block)
    #
    # print(mha_out_block.shape)
    # print(mha_attn_block.shape, mha_attn_block[0][0].numpy())  # attention within each attn block

    # (N, H, Tq, Tv) -> (H, N, Tq, Tv) ->
    # mha_attn_block_recover = tf.reshape(tf.transpose(mha_out_block, (1, 0, 2, 3)), [-1, tf.shape(query)[0], L, ]

    # fmha = FactorizedAttention(ctc_len=16, num_heads=4, d_model=48, blocks=4)
    #
    # fmha_out, fmha_attn = fmha.prev_row_attn(query, key, value)
    #
    # print(fmha_out.shape, fmha_attn.shape)
    #
    # print(fmha_attn[0][0].numpy())
    #
    # fmha_col, fmha_col_attn = fmha.col_attn(query, key, value)
    #
    # print(fmha_col.shape, fmha_col_attn.shape)
    #
    # print(fmha_col_attn[0][0].numpy())
    #
    # fmha_row, fmha_row_attn = fmha.row_attn(query, key, value)
    #
    # print(fmha_row.shape, fmha_row_attn.shape)
    #
    # print(fmha_row_attn[0][0].numpy())

    for attn_func in [0, 1, 2]:
        fmha = FactorizedAttention(ctc_len=16, num_heads=4, d_model=48, blocks=4, attn_func=attn_func)
        inputs = tf.random.normal([4, 16, 48])
        fmha_out, fmha_attn_w = fmha(inputs, training=False, return_attention_weights=True)

        print("Attention Type: ", fmha.attn_type)
        print(fmha_out.shape, fmha_attn_w.shape)

        print(fmha_attn_w[0][0].numpy())

        # Check Sampling
        sample_x = tf.random.normal([4, 1, 48])
        sample_out, sample_attn_w = fmha(sample_x, traininig=False, return_attention_weights=True)
        print(sample_out.shape, sample_attn_w.shape)


