import sys
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from src.transformer.multi_head_attention import positional_encoding, PositionalEmbedding
from src.transformer.transformer import FactorizedTransformer


class FMHABasedAutoregressiveModel(keras.Model):
    """
          @latent_dim
          @levels: number of stacked VQ-VAE (could be hierarchical like VQ-VAE-2, or independent like jukebox paper proposed)
        """

    def __init__(self,
                 target_vocab_size,  # num categorical distribution
                 width,  # d_model, model width
                 depth,  # num of ResidualAttention blocks
                 blocks,  # Height of 2D attention blocks, the width is decided by context_length
                 m_attn=0.25,
                 m_mlp=1.0,
                 heads=1,
                 attn_stacks=1,
                 maximum_pos_encoding=5000,
                 drop_out_rate=0.1,
                 context_length=None,  # Almost not needed for vanilla Attention
                 pos_emb=True,
                 **kwargs):
        super(FMHABasedAutoregressiveModel, self).__init__(**kwargs)

        self.context_length = tf.reduce_prod(context_length)  # (T, )
        self.bins = target_vocab_size
        self.d_model = width
        self.depth = depth
        self.use_pos_embedding = pos_emb

        # Start Token
        self.start_token = self.bins - 1  # TODO: temporary, for |zq| == 512, pass in 513

        # 1. Embedding map indices to d_model length embedding
        self.x_embedding = layers.Embedding(self.bins, self.d_model)
        # 2.a Constant Positional Embedding to add timing info (TODO: is this applicable?)
        self.x_pos_encoding = positional_encoding(maximum_pos_encoding,
                                                  self.d_model)  # (1, maximum_pos_encoding, d_model)
        # 2.b Explicit positional Embedding
        if self.use_pos_embedding:
            self.x_pos_embedding = PositionalEmbedding(self.context_length, self.d_model)
        # 3. Factorized Transformer
        self.transformer = FactorizedTransformer(ctc_len=self.context_length, num_heads=heads, d_model=self.d_model,
                                                 blocks=blocks, attn_stacks=attn_stacks, depth=depth, m_attn=m_attn,
                                                 m_mlp=m_mlp, rate=drop_out_rate)

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

        x = self.x_embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # x += self.x_pos_embedding.get_embeddings(seq_len) # (1, seq_len, d_model)
        if self.use_pos_embedding:
            x += self.x_pos_embedding(x, training=training)  # (1, seq_len, d_model)
        else:
            x += self.x_pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        x, attention_weights = self.transformer(x, training=training)

        # x.shape == (batch_size, target_seq_len, d_model)

        # (N, T, D)
        final_output = self.out(x)

        return final_output, attention_weights

    @tf.function
    def sample(self, n_samples, max_length=None, return_attention_weights=False):
        """
        TODO: need to support sampling for factorized attention first...
        :param n_samples:
        :param max_length:
        :param return_attention_weights:
        :return:
        """
        tf.print(f'[DEBUG SAMPLING] Random Gumbel Noise Sample at each step', output_stream=sys.stdout)
        # tf.print(f'[DEBUG SAMPLING] Greedy Sample at each step', output_stream=sys.stdout)
        if max_length is None:
            max_length = self.context_length

        start = tf.constant(self.start_token, dtype=tf.int64, shape=[n_samples, ])

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())  # (N, i+1); when i==0, start_token at each position
            # (N, i+1, D) raw logits; Note that we don't need to do full sequence inference... TODO: improvement
            predictions, _ = self.call(output, training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (N, 1, D)

            # Sampling 1: Greedy Search
            ## TODO: others?
            # predicted_id = tf.argmax(predictions, axis=-1)  # (N, 1)

            # Sampling 2: Sample from Categorical distribution
            ##
            # dist = tfp.distributions.Categorical(logits=predictions)
            # predicted_id = tf.cast(dist.sample(), dtype=tf.int64) # (N, 1)

            # Sampling 2.5: Gumbel Noise
            dist = tfp.distributions.RelaxedOneHotCategorical(1, logits=predictions)
            gumbel_samples = dist.sample()  # (N, 1, D)
            # tf.print(f"[DEBUG] Gumbel output shape: {gumbel_samples.shape}", output_stream=sys.stdout)
            predicted_id = tf.argmax(gumbel_samples, axis=-1)  # (N, 1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i + 1, tf.squeeze(predicted_id, axis=-1)) # (N, )

            # if predicted_id == end:
            #     break

        #  output.shape (N, max_length+1)
        output = tf.transpose(output_array.stack())

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        if return_attention_weights:
            _, attention_weights = self.call(output[:, :-1], training=False)
            return output, attention_weights

        return output

    def random_sample(self, loss_fn, seq_length=None, iterations=10, batch_per_iter=4, token_freq=0.50):
        """
        Random Search
        :param loss_fn:
        :param seq_length:
        :param iterations:
        :return:
        """
        best_loss = 1e10
        best_sample = tf.zeros(shape=[1, seq_length])

        if seq_length is None:
            seq_length = self.context_length

        for i in tf.range(iterations):
            # (N, T+1) including the 'start_token' at 0
            sampled_out = self.sample(n_samples=batch_per_iter, max_length=seq_length, return_attention_weights=False)
            # (N, T)
            target = sampled_out[:, 1:]
            # (N, T, V); V being the codebook size
            raw_logits, _ = self.call(sampled_out[:, :-1], training=False)
            # compute the loss
            # (N, T)
            loss = loss_fn(target, raw_logits)
            loss = tf.reduce_mean(loss, axis=-1) # (N, )
            # best_sample_idx = tf.argmin(loss)
            # best_cur_iter_loss = tf.reduce_min(loss)
            # tf.print(f'Best Loss: {best_cur_iter_loss} for Iteration: {i}', output_stream=sys.stdout)
            # if best_cur_iter_loss < best_loss:
            #     # $$$Avoid too much occurrence of single token...
            #     best_cur_sample = sampled_out[best_sample_idx]
            #     y, idx, count = tf.unique_with_counts(best_cur_sample)
            #     if tf.reduce_max(count) >= seq_length // 2:
            #         tf.print(f'Token {y[tf.argmax(count)]} occurred {tf.reduce_max(count)} Times! Skipping...')
            #     else:
            #         best_loss = best_cur_iter_loss
            #         best_sample = best_cur_sample
            sorted_loss_idx = tf.argsort(loss, direction='ASCENDING')
            # sorted_loss = tf.gather(loss, sorted_loss_idx)
            iter_done = False
            for k in tf.range(len(sorted_loss_idx)):
                cur_idx = sorted_loss_idx[k]
                cur_loss = loss[cur_idx]
                # tf.print(f'Best Loss: {cur_loss} for Iteration: {i}-{k}', output_stream=sys.stdout)
                if cur_loss < best_loss:
                    tf.print(f'Best Loss: {cur_loss} for Iteration: {i}-{k}', output_stream=sys.stdout)
                    # $$$Avoid too much occurrence of single token...
                    best_cur_sample = sampled_out[cur_idx]
                    y, idx, count = tf.unique_with_counts(best_cur_sample)
                    if tf.reduce_max(count) >= tf.cast(seq_length * token_freq, tf.int32):
                        tf.print(f'Token {y[tf.argmax(count)]} occurred {tf.reduce_max(count)} Times! Skipping...', output_stream=sys.stdout)
                    else:
                        best_loss = cur_loss
                        best_sample = best_cur_sample
                        # iter_done = True
                else:
                    break


        return best_sample, best_loss




if __name__ == '__main__':
    print('Factorized Multi-Head Attention based Autoregressive module!')

    sample_in = tf.random.uniform((32, 16), dtype=tf.int64, minval=0, maxval=200)  # (N, T)

    automha = FMHABasedAutoregressiveModel(context_length=sample_in.shape[1:],
                                           target_vocab_size=512,
                                           width=128,
                                           depth=6,
                                           heads=2,
                                           blocks=4,
                                           attn_stacks=1)
    # automha.compile(run_eagerly=True)

    out, attn_w = automha(sample_in, training=False)

    automha.summary()
    # print(automha.x_pos_embedding.trainable_variables)

    print("Autoregressive Model Output: ", out.shape)
    print("Multi-head Self Attention: ", {k: v.shape for k, v in attn_w.items()})

    for _, v in attn_w.items():
        print(v.numpy()[0][-1])

    # Test Sampling
    ## attention weights for the whole sampled batch
    print(f"Validate Sampling...")
    ## 1. sample_size==1 2. sample_size > 1
    sampled_sequence, sample_attn_w = automha.sample(n_samples=4, return_attention_weights=True)

    print(sampled_sequence.shape)
    print(sampled_sequence)

    for v, attn in sample_attn_w.items():
        print(v)
        print(attn[0][0])  # (One Head, one sampled

    print(f'Validate Random Search...')
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    best_sample, best_loss = automha.random_sample(loss_fn, seq_length=10, iterations=10)

    print(f'Best Sample: {best_sample.shape}, with loss per word: {best_loss}')


    ## Idx Sort + Gather example
    b = tf.random.normal([6, 4])
    idx = tf.argsort(tf.reduce_sum(b, axis=-1))
    tf.gather(b, idx, batch_dims=0)

