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
        print(f'[DEBUG] CAUSAL CONV1D, with No Mask pre-row attention')
        # print(f'[DEBUG] DENSE QKV')
        # self.qkv_conv = layers.Conv1D(self.width * 3, 3, strides=1, dilation_rate=1, padding="same") # Converge to 99% accuracty < 30 epochs (training, ground-truth fed each step)
        self.qkv_conv = layers.Conv1D(self.width * 3, 3, strides=1, dilation_rate=1, padding="causal")  # CAUSAL CONV1D
        # self.qkv_conv = layers.Dense(self.width * 3) # For Debug only, given the conv1D is $$NON-CAUSAL$$$!!!
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

        x = self.qkv_conv(inputs, training=training)  # (N, T, 3*d)
        query, key, value = tf.split(x, num_or_size_splits=3, axis=-1)  # (N, T, d)

        attn_out, attn_w = self.attn(query, key, value, training=training)
        out = self.proj(attn_out, training=training)  # (N, T, D)

        if return_attention_weights:
            return self.attn_dropout(out, training=training), attn_w
        return self.attn_dropout(out, training=training)

    def row_attn(self, q, k, v, training=False, sample=False):
        """

        :param training:
        :param q:
        :param k:
        :param v:
        :param sample: TODO
        :return:
        """

        N, L, D = shape_list(k)
        Lq = shape_list(q)[1]

        # tf.print("Query Length: ", Lq)
        # tf.print("Block Length: ", self.block_len)
        # TODO
        # assert Lq == L
        tf.debugging.assert_equal(
            Lq, L, message="Query length and Key/Value length not equal!",
            summarize=None, name=None
        )

        # print(f'[DEBUG][Factorized Attn|Row] If-condition in TF-Function...')
        trail_len = Lq % self.block_len
        num_blocks = Lq // self.block_len  # current full blocks

        # For tf.function
        # TODO: understand this...
        mha_out = q
        mha_attn = tf.zeros([N, self.num_heads, Lq, L])

        if trail_len > 0:
            q_cur = q[:, -trail_len:, :]
            k_cur = k[:, -trail_len:, :]
            v_cur = v[:, -trail_len:, :]
            # [N, trail_len, D]
            mha_out, mha_attn = self.mha(query=q_cur, key=k_cur, value=v_cur,
                                             attention_mask=create_look_ahead_mask(trail_len, trail_len),
                                             return_attention_scores=True, training=training)

            q = q[:, :-trail_len, :]
            k = k[:, :-trail_len, :]
            v = v[:, :-trail_len, :]

        # for pixels (if any) before latest block, regular row attention
        if num_blocks > 0:
            q = tf.reshape(q, [N * num_blocks, self.block_len, D])
            k = tf.reshape(k, [N * num_blocks, self.block_len, D])
            v = tf.reshape(v, [N * num_blocks, self.block_len, D])

            attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
            mha_out_complete, mha_attn_complete = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
                                                  return_attention_scores=True)

            mha_out_complete = tf.reshape(mha_out_complete, [N, num_blocks * self.block_len, D])

            if trail_len > 0:
                mha_out = tf.concat([mha_out_complete, mha_out], axis=1)
            else:
                mha_out = mha_out_complete
                mha_attn = mha_attn_complete

        # tf.debugging.assert_equal(
        #     tf.shape(mha_out), tf.constant([N, Lq, D]), message="Shape not Legit during sampling",
        #     summarize=None, name=None
        # )
        return mha_out, mha_attn

        # # TODO: solve this... either work for training or work only for sampling....
        # if not sample:
        #     num_blocks = Lq // self.block_len
        #
        #     tf.debugging.assert_equal(Lq, num_blocks * self.block_len, "WTF...")
        #
        #     # reshape the input sequence (N, L, D) into 2D sequence [blocks, block length]
        #     q = tf.reshape(q, [N * num_blocks, self.block_len, D])
        #     # TODO: if Lq != L, then the batch dim would be different!
        #     k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        #     v = tf.reshape(v, [N * num_blocks, self.block_len, D])
        #
        #     attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        #     mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
        #                                  return_attention_scores=True, training=training)
        #
        #     return tf.reshape(mha_out, [N, Lq, D]), mha_attn
        # else:
        #     if Lq % self.block_len == 0:
        #     # if tf.shape(q)[1] % self.block_len == 0:
        #     # if tf.equal(Lq % self.block_len, 0):
        #     # if tf.math.floormod(Lq, self.block_len) == tf.constant(0):
        #         num_blocks = Lq // self.block_len
        #
        #         # reshape the input sequence (N, L, D) into 2D sequence [blocks, block length]
        #         q = tf.reshape(q, [N * num_blocks, self.block_len, D])
        #         # TODO: if Lq != L, then the batch dim would be different!
        #         k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        #         v = tf.reshape(v, [N * num_blocks, self.block_len, D])
        #
        #         attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        #         mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
        #                                      return_attention_scores=True, training=training)
        #
        #         return tf.reshape(mha_out, [N, Lq, D]), mha_attn
        #     # During Autoregressive sampling steps, when generation has to be done steps by steps
        #     else:
        #         # for pixels ( < block_len) within latest block, apply mha directly
        #         trail_len = Lq % self.block_len
        #         tf.debugging.assert_none_equal(trail_len, 0, "WTF...")
        #         num_blocks = Lq // self.block_len  # current full blocks
        #         q_cur = q[:, -trail_len:, :]
        #         k_cur = k[:, -trail_len:, :]
        #         v_cur = v[:, -trail_len:, :]
        #         # [N, trail_len, D]
        #         mha_out, mha_attn_cur = self.mha(query=q_cur, key=k_cur, value=v_cur,
        #                                          attention_mask=create_look_ahead_mask(trail_len, trail_len),
        #                                          return_attention_scores=True, training=training)
        #         # for pixels (if any) before latest block, regular row attention
        #         if num_blocks > 0:
        #             q = tf.reshape(q[:, :-trail_len, :], [N * num_blocks, self.block_len, D])
        #             k = tf.reshape(k[:, :-trail_len, :], [N * num_blocks, self.block_len, D])
        #             v = tf.reshape(v[:, :-trail_len, :], [N * num_blocks, self.block_len, D])
        #
        #             attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        #             mha_out_complete, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
        #                                                   return_attention_scores=True)
        #
        #             mha_out_complete = tf.reshape(mha_out_complete, [N, num_blocks * self.block_len, D])
        #             mha_out = tf.concat([mha_out_complete, mha_out], axis=1)
        #
        #         tf.debugging.assert_equal(
        #             tf.shape(mha_out), tf.constant([N, Lq, D]), message="Shape not Legit during sampling",
        #             summarize=None, name=None
        #         )
        #         return mha_out, mha_attn_cur

    def col_attn(self, q, k, v, training=False, sample=False):
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
        # assert Lq == L
        tf.debugging.assert_equal(
            Lq, L, message="Query length and Key/Value length not equal!",
            summarize=None, name=None
        )

        trail_len = Lq % self.block_len
        num_blocks = Lq // self.block_len  # current full blocks

        # TODO: Eliminate all if-conditions ...
        mha_out = tf.random.normal([N, 0, D]) # Empty
        mha_attn = tf.zeros([N, self.num_heads, Lq, L])

        if trail_len > 0:
            k_cur = k[:, -trail_len:, :] # (N, trail_l, D)
            k_cur_prev = tf.reshape(k[:, :-trail_len, :], [N, num_blocks, self.block_len, D])[:, :, :trail_len, :] # (N, blocks, trail_l, D)
            k_cur = tf.concat([k_cur_prev, tf.expand_dims(k_cur, axis=1)], axis=1)
            v_cur = v[:, -trail_len:, :]
            v_cur_prev = tf.reshape(v[:, :-trail_len, :], [N, num_blocks, self.block_len, D])[:, :, :trail_len, :] # (N, blocks, trail_l, D)
            v_cur = tf.concat([v_cur_prev, tf.expand_dims(v_cur, axis=1)], axis=1)
            k_cur = tf.reshape(tf.transpose(k_cur, [0, 2, 1, 3]), [N*trail_len, num_blocks+1, D]) # (N*trail_l, blocks+1, D)
            v_cur = tf.reshape(tf.transpose(v_cur, [0, 2, 1, 3]), [N * trail_len, num_blocks + 1, D])  # (N*trail_l, blocks+1, D)

            q_cur = q[:, -trail_len:, :] # (N, trail_l, D)
            q_cur = tf.reshape(q_cur, [N*trail_len, 1, D]) # (N*trail_l, 1, D)
            # q_cur = tf.reshape(q[:, :-trail_len, :], [N, num_blocks, self.block_len, D])[:, :, :trail_len, :] # (N, blocks, trail_l, D)

            # No Masks needed
            # out: (N*trail_l, 1, D); attn: (N*trail_l, H, 1, blocks/rows+1)
            mha_out, mha_attn = self.mha(query=q_cur, key=k_cur, value=v_cur, attention_mask=None,
                                         return_attention_scores=True, training=training)
            mha_out = tf.reshape(mha_out, [N, trail_len, D])

            q = q[:, :-trail_len, :]
            k = k[:, :-trail_len, :]
            v = v[:, :-trail_len, :]

        q = tf.transpose(tf.reshape(q, [N, num_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)
        # print("After Col Transpose to Row: ", q[0, ..., 0])
        k = tf.transpose(tf.reshape(k, [N, num_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)
        v = tf.transpose(tf.reshape(v, [N, num_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)

        q = tf.reshape(q, [N * self.block_len, num_blocks, D])  # (N*l, blocks, D), l*blocks = Tq - trail_l
        k = tf.reshape(k, [N * self.block_len, num_blocks, D])
        v = tf.reshape(v, [N * self.block_len, num_blocks, D])

        attn_mask = create_look_ahead_mask(num_blocks, num_blocks)

        # out: (N*l, blocks, D)
        # Note if num_blocks == 0 (sampling in the 1st very block...), out: (N*l, 0, D)
        mha_out_complete, mha_attn_complete = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
                                     return_attention_scores=True, training=training)
        # $$$ Note we need to transpose back to input sequence order
        mha_out_complete = tf.reshape(mha_out_complete, [N, self.block_len,  num_blocks, D])
        mha_out_complete = tf.transpose(mha_out_complete, [0, 2, 1, 3]) # (N, blocks, l, D)
        mha_out_complete = tf.reshape(mha_out_complete, [N, num_blocks * self.block_len, D])

        if trail_len > 0:
            mha_out = tf.concat([mha_out_complete, mha_out], axis=1) # (N, Lq, D)
        else:
            mha_out = mha_out_complete
            mha_attn = mha_attn_complete

        return mha_out, mha_attn

        # # Training Version (Full size Attn Block)
        # num_q_blocks = Lq // self.block_len
        # num_kv_blocks = L // self.block_len  # q, kv num blocks no need to be equal
        # # print("Before Col Transpose to Row: ", tf.reshape(q, [N, num_q_blocks, self.block_len, D])[0,...,0])
        # q = tf.transpose(tf.reshape(q, [N, num_q_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)
        # # print("After Col Transpose to Row: ", q[0, ..., 0])
        # k = tf.transpose(tf.reshape(k, [N, num_kv_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)
        # k = tf.transpose(tf.reshape(v, [N, num_kv_blocks, self.block_len, D]), [0, 2, 1, 3])  # (N, l, blocks, D)
        #
        # q = tf.reshape(q, [N * self.block_len, num_q_blocks, D])  # (N*l, blocks, D), l*blocks = Tq
        # k = tf.reshape(k, [N * self.block_len, num_kv_blocks, D])
        # v = tf.reshape(v, [N * self.block_len, num_kv_blocks, D])
        #
        # attn_mask = create_look_ahead_mask(num_q_blocks, num_kv_blocks)
        #
        # mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
        #                              return_attention_scores=True , training=training)
        #
        # return tf.reshape(mha_out, [N, Lq, D]), mha_attn

    def prev_row_attn(self, q, k, v, training=False, sample=False):
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
        # assert Lq == L
        tf.debugging.assert_equal(
            Lq, L, message="Query length and Key/Value length not equal!",
            summarize=None, name=None
        )

        trail_len = Lq % self.block_len
        num_blocks = Lq // self.block_len  # current full blocks

        # TODO: understand this...
        mha_out = q
        mha_attn = tf.zeros([N, self.num_heads, Lq, L])

        if trail_len > 0:
            q_cur = q[:, -trail_len:, :] # (N, trail_l, D)
            # Decide prev key and value
            if num_blocks > 0:
                start_idx = (num_blocks-1) * self.block_len
                k_cur = k[:, start_idx:start_idx+self.block_len, :] # (N, l, D)
                v_cur = v[:, start_idx:start_idx+self.block_len, :] # (N, l, D)
            else:
                # Pad with zero
                k_cur = tf.zeros([N, self.block_len, D]) # (N, l, D)
                v_cur = tf.zeros([N, self.block_len, D]) # (N, l, D)

            # No Masks needed
            # out: (N, trail_l, D); attn: (N, H, trail_l, l)
            mha_out, mha_attn = self.mha(query=q_cur, key=k_cur, value=v_cur, attention_mask=None,
                                         return_attention_scores=True, training=training)

            q = q[:, :-trail_len, :]
            k = k[:, :-trail_len, :]
            v = v[:, :-trail_len, :]

        q = tf.reshape(q, [N * num_blocks, self.block_len, D]) # (N*blocks, l, D)

        # 1.remove current block from bottom
        # print("Before Shift Blocks: ", tf.reshape(k, [N, num_blocks, self.block_len, D])[0, ..., 0].numpy())
        # 1. For sampling with empty prev row: Pad first
        k = tf.reshape(k, [N, num_blocks, self.block_len, D]) # (N, blocks, l, D)
        # print("After Shift Blocks: ", k[0, ..., 0].numpy())
        v = tf.reshape(v, [N, num_blocks, self.block_len, D])
        # 2. pad dim:-3, the num_blocks dimension with extra block at top
        # print("Before Pad: ", k[0, ..., 0].numpy())
        k = tf.pad(k, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # print("After Pad: ", k[0, ..., 0].numpy())
        v = tf.pad(v, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # Then Shift forward to remove current block
        k = k[:, :-1, :, :]
        v = v[:, :-1, :, :]
        # 1 && 2 combined ==> shift forward k, v blocks by one block
        k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        v = tf.reshape(v, [N * num_blocks, self.block_len, D])
        # TODO: if Lq < L?

        # No Masks needed
        # out: (N, blocks*l, D)
        mha_out_complete, mha_attn_complete = self.mha(query=q, key=k, value=v, attention_mask=None,
                                     return_attention_scores=True, training=training)
        mha_out_complete = tf.reshape(mha_out_complete, [N, num_blocks*self.block_len, D])

        if trail_len > 0:
            mha_out = tf.concat([mha_out_complete, mha_out], axis=1) # (N, Lq, D)
        else:
            mha_out = mha_out_complete
            mha_attn = mha_attn_complete

        return mha_out, mha_attn

        # num_blocks = Lq // self.block_len
        #
        # q = tf.reshape(q, [N * num_blocks, self.block_len, D])
        #
        # # 1.remove current block from bottom
        # # print("Before Shift Blocks: ", tf.reshape(k, [N, num_blocks, self.block_len, D])[0, ..., 0].numpy())
        # k = tf.reshape(k, [N, num_blocks, self.block_len, D])[:, :-1, :, :]
        # # print("After Shift Blocks: ", k[0, ..., 0].numpy())
        # v = tf.reshape(v, [N, num_blocks, self.block_len, D])[:, :-1, :, :]
        # # 2. pad dim:-3, the num_blocks dimension with extra block at top
        # # print("Before Pad: ", k[0, ..., 0].numpy())
        # k = tf.pad(k, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # # print("After Pad: ", k[0, ..., 0].numpy())
        # v = tf.pad(v, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]], mode='CONSTANT')
        # # 1 && 2 combined ==> shift forward k, v blocks by one block
        # k = tf.reshape(k, [N * num_blocks, self.block_len, D])
        # v = tf.reshape(v, [N * num_blocks, self.block_len, D])
        # # TODO: if Lq < L?
        #
        # attn_mask = create_look_ahead_mask(self.block_len, self.block_len)
        # mha_out, mha_attn = self.mha(query=q, key=k, value=v, attention_mask=attn_mask,
        #                              return_attention_scores=True, training=training)
        #
        # return tf.reshape(mha_out, [N, Lq, D]), mha_attn


if __name__ == '__main__':
    print("Factorized Attention Module!")

    np.set_printoptions(precision=6)

    # Sequence of length 16
    query = tf.random.normal([4, 16, 48])
    key = tf.random.normal([4, 16, 48])
    value = tf.random.normal([4, 16, 48])

    for attn_func in [0, 1, 2]:
    # for attn_func in [0]: # ROW
    # for attn_func in [1]: # COL
    # for attn_func in [2]: # PREV-ROW
    #     fmha = FactorizedAttention(ctc_len=16, num_heads=4, d_model=48, blocks=4, attn_func=attn_func)
        fmha = FactorizedAttention(ctc_len=16, num_heads=1, d_model=4, blocks=4, attn_func=attn_func)
        # QKV Conv1D

        inputs = tf.random.normal([4, 16, 48])
        fmha_out, fmha_attn_w = fmha(inputs, training=False, return_attention_weights=True)

        print(f"Conv 1D variables: {print([(v.name, w.shape) for v, w in zip(fmha.qkv_conv.trainable_variables, fmha.qkv_conv.get_weights())])}")

        print(f"------------------------------Attention Type: {fmha.attn_type} ------------------------------------")
        print(fmha_out.shape, fmha_attn_w.shape)

        print(fmha_attn_w[0][0].numpy())

        # Check Sampling

        print("Start Sampling Test....")
        for i in range(16): # until batch input sequence len
            # pre-condition
            sample_x = inputs[:, :i+1, :]
            tf.debugging.assert_equal(sample_x, inputs[:, :i+1, :])
            sample_out, sample_attn_w = fmha(sample_x, traininig=False, return_attention_weights=True, sample=True)
            print(sample_out.shape, sample_attn_w.shape)
            # print(f"Sample {i}")
            # print(sample_out[..., 0])
            # print(f"Batch {i}")
            # print(fmha_out[:, :i + 1, :][..., 0])

            # sample output at step i should equal to the same position batch output
            diff = tf.math.abs(sample_out - fmha_out[:, :i + 1, :])

            tf.debugging.assert_less_equal(tf.reduce_max(diff), 1e-6)
            print(f"Sample {i}, Max DIFF: {tf.reduce_max(diff)}")
            # tf.debugging.assert_equal(sample_out, fmha_out[:, :i+1, :])
            # print(sample_attn_w[0][0])
            # Get The latest token only... to be optimized (don't need to full pass the whole generated sequence at each step...)
            # sample_x = tf.concat([sample_x, sample_out[:, -1:, :]], axis=1
            # break down inputs at each step

        fmha_out

        # # Check Sampling
        # # sample_x = tf.random.normal([4, 1, 48])
        # sample_x = inputs[:, :1, :]

        # # Attn function level
        # x = tf.random.normal([4, 16, 12])
        #
        # batch_out, batch_attn_w = fmha.attn(x, x, x, training=False)
        # print(batch_out.shape)
        #
        # sample_x = x[:, :1, :]
        #
        # for i in range(16):
        #     # pre condition
        #     sample_x = x[:, :i+1, :]
        #     tf.debugging.assert_equal(sample_x, x[:, :i+1, :])
        #     sample_out, sample_attn_w = fmha.attn(sample_x, sample_x, sample_x, training=False)
        #     print(sample_out.shape, sample_attn_w.shape)
        #
        #     diff = tf.math.abs(sample_out - batch_out[:, :i + 1, :])
        #
        #     # check the output
        #     tf.debugging.assert_less_equal(tf.reduce_max(diff), 1e-6)

            # attach next


        # for i in range(16): # until batch input sequence len
        #     # pre-condition
        #     sample_x = inputs[:, :i+1, :]
        #     tf.debugging.assert_equal(sample_x, inputs[:, :i+1, :])
        #     sample_out, sample_attn_w = fmha(sample_x, traininig=False, return_attention_weights=True, sample=True)
        #     print(sample_out.shape, sample_attn_w.shape)
        #
        #     # sample output at step i should equal to the same position batch output
        #     diff = tf.math.abs(sample_out - fmha_out[:, :i + 1, :])
        #
        #     tf.debugging.assert_less_equal(tf.reduce_max(diff), 1e-6)
        #     # tf.debugging.assert_equal(sample_out, fmha_out[:, :i+1, :])
        #     # print(sample_attn_w[0][0])
        #     # Get The latest token only... to be optimized (don't need to full pass the whole generated sequence at each step...)
        #     # sample_x = tf.concat([sample_x, sample_out[:, -1:, :]], axis=1
        #     # break down inputs at each step
        #
        # fmha_out
