import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.autoregressive.autoregressive_fmha import FMHABasedAutoregressiveModel


class VQVAESampler(keras.models.Model):
    def __init__(self, down_depth, strides, n_ctxs, **kwargs):
        super(VQVAESampler, self).__init__(**kwargs)

        self.downsamples = [stride ** down for stride, down in zip(strides, down_depth)]
        # self.hop_lengths = tf.math.cumprod(self.downsamples)  # hop length for each level
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = len(down_depth)

        rescale_zs_shapes = lambda level, cur_level: (n_ctxs[cur_level] * self.hop_lengths[cur_level] // self.hop_lengths[level], )

        self.priors = []

        self.x_cond_kwargs = dict(dilation_factor=3, dilation_cycle=4, residual_width=32, residual_depth=8)

        for l in range(self.levels):
            zs_shapes = [rescale_zs_shapes(l_, l) for l_ in range(self.levels)]

            # Context length for current level prior
            assert zs_shapes[l][0] == n_ctxs[l]

            print(f"[DEBUG] Scaled Context lengths for current level: {zs_shapes}")

            # TODO: Move to Prior
            x_cond_kwargs = None
            if l != self.levels - 1:
                x_cond_kwargs = self.x_cond_kwargs

            prior = FMHABasedAutoregressiveModel(context_length=zs_shapes[l],
                                                 target_vocab_size=512,
                                                 width=128,
                                                 depth=6,
                                                 heads=2,
                                                 blocks=4,
                                                 attn_stacks=1,
                                                 zq_shapes=zs_shapes,  # [(16,), (4,)],
                                                 level=l,
                                                 levels=self.levels,
                                                 downs=down_depth,
                                                 strides=strides,
                                                 cond_kwargs=x_cond_kwargs)

            self.priors.append(prior)

    def sample(self, n_samples):
        """
        From Top to Bottom
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

            print(f"[DEBUG] Upper level Sampled Codes: {x_cond}")


            sampled_sequence, sample_attn_w = self.priors[level].sample(n_samples=n_samples,
                                                                        return_attention_weights=True,
                                                                        x_cond=x_cond)  # condition on upper-level (if exists)
            # TODO: remove start/prime token
            zs[level] = tf.concat([zs[level], sampled_sequence], axis=-1)

        print(f"[DEBUG] Sampled Sequence for all levels: \n {zs}")

        return NotImplementedError

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
    sampler = VQVAESampler(down_depth=[3, 2, 2], strides=[2, 2, 2], n_ctxs=[16, 16, 16])

    sampler.sample(n_samples=3)
