import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.autoregressive.autoregressive_fmha import FMHABasedAutoregressiveModel
from prior import Prior


class VQVAESampler(keras.models.Model):
    def __init__(self, down_depth, strides, n_ctxs, codebook_size=513, priors=None, num_genres=None, **kwargs):
        super(VQVAESampler, self).__init__(**kwargs)

        self.downsamples = [stride ** down for stride, down in zip(strides, down_depth)]
        # self.hop_lengths = tf.math.cumprod(self.downsamples)  # hop length for each level
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = len(down_depth)
        self.bins = codebook_size

        rescale_zs_shapes = lambda level, cur_level: (
        n_ctxs[cur_level] * self.hop_lengths[cur_level] // self.hop_lengths[level],)

        self.priors = []

        self.x_cond_kwargs = dict(dilation_factor=3, dilation_cycle=4, residual_width=32, residual_depth=8)
        self.prior_kwargs = dict(width=128, depth=6, heads=2, blocks=4, attn_stacks=1, drop_out_rate=0.0)

        if priors is not None:
            assert len(priors) == self.levels
            self.priors = priors
        else:
            for l in range(self.levels):
                # TODO: restore from checkpoint...
                zs_shapes = [rescale_zs_shapes(l_, l) for l_ in range(self.levels)]

                # Context length for current level prior
                assert zs_shapes[l][0] == n_ctxs[l]

                print(f"[DEBUG] Scaled Context lengths for current level: {zs_shapes}")

                # TODO: Move to Prior
                x_cond_kwargs = None
                if l != self.levels - 1:
                    x_cond_kwargs = self.x_cond_kwargs

                # prior = FMHABasedAutoregressiveModel(context_length=zs_shapes[l],
                #                                      target_vocab_size=512,
                #                                      width=128,
                #                                      depth=6,
                #                                      heads=2,
                #                                      blocks=4,
                #                                      attn_stacks=1,
                #                                      zq_shapes=zs_shapes,  # [(16,), (4,)],
                #                                      level=l,
                #                                      levels=self.levels,
                #                                      downs=down_depth,
                #                                      strides=strides,
                #                                      cond_kwargs=x_cond_kwargs)

                prior = Prior(level=l,
                              z_shapes=zs_shapes,
                              bins=self.bins,
                              down_depth=down_depth,
                              strides=strides,
                              prior_kwargs=self.prior_kwargs,
                              x_cond_kwargs=x_cond_kwargs,
                              vqvae_model=None,
                              genre_classes=num_genres)

                self.priors.append(prior)

    def sample(self, n_samples, y_genre=None):
        """
        From Top to Bottom
        :param n_samples
        :param y_genre TODO: multiple labels...
        :return:
        """
        zs = [tf.zeros([n_samples, 0], dtype=tf.int64) for _ in range(self.levels)]

        print(zs)

        for level in reversed(range(self.levels)):
            """
            From Top to Bottom
            """

            start = 0
            end = self.priors[level].context_length

            print(f"[DEBUG] Sampling at level {level}, [{start}, {end})")

            x_cond = self.priors[level].get_cond(zs, start, end)

            print(f"[DEBUG] Upper level Sampled Codes: {x_cond.shape if x_cond is not None else x_cond}, {x_cond}")

            sampled_sequence, sample_attn_w = self.priors[level].sample(n_samples=n_samples,
                                                                        return_attn_weights=True,
                                                                        z_cond=x_cond,
                                                                        # condition on upper-level (if exists)
                                                                        y=y_genre)  # Condition on genre classes
            # TODO: remove start/prime token
            # zs[level] = tf.concat([zs[level], sampled_sequence], axis=-1)
            ## REMOVE start Token
            zs[level] = tf.concat([zs[level], sampled_sequence[:, 1:]], axis=-1)

        print(f"[DEBUG] Sampled Sequence for all levels: \n {zs}")

        return zs

    def sample_level(self, zs, level):
        """
        Sample a single window of length=n_ctx at level=level

        :param level:
        :param zs:
        :return:
        """
        return NotImplementedError


if __name__ == '__main__':
    print('Prior Sampler module!')

    # sampler = VQVAESampler(down_depth=[3, 2, 2], strides=[2, 2, 2], n_ctxs=[8192, 8192, 6144])
    # sampler = VQVAESampler(down_depth=[3, 2], strides=[2, 2], n_ctxs=[880, 880])
    # sampler = VQVAESampler(down_depth=[3, 2, 2], strides=[2, 2, 2], n_ctxs=[16, 16, 16])
    sampler = VQVAESampler(down_depth=[3, 2, 2], strides=[2, 2, 2], n_ctxs=[64, 16, 4])

    sampler.sample(n_samples=3)

    print('======= Sample with Genre Labels...')
    y = tf.constant([3, 2, 1])
    y_cond_sampler = VQVAESampler(down_depth=[3, 2, 2], strides=[2, 2, 2], n_ctxs=[64, 16, 4], num_genres=10)

    y_cond_sampler.sample(n_samples=3, y_genre=y)
