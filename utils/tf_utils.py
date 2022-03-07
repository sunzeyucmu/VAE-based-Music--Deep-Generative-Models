import os, sys, time
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import SAMPLE_RATE, IDX_TO_GENRES


def compare_t(t1, t2):
    not_equal = tf.cast(t1 != t2, tf.float32)
    return tf.reduce_sum(not_equal), not_equal


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def plot_attention_head(attention_mat, in_tokens=None, translated_tokens=None):
    """
      The plot is of the attention when a token was generated.
      The model didn't generate `<START>` in the output. Skip it.
      translated_tokens = translated_tokens[1:]
    """

    ax = plt.gca()
    ax.matshow(attention_mat)
    # ax.set_xticks(range(len(in_tokens)))
    # ax.set_yticks(range(len(translated_tokens)))

    # labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    # ax.set_xticklabels(
    #     labels, rotation=90)

    # labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    # ax.set_yticklabels(labels)


def plot_attention_weights(attention_heads, name="MultiHeadAttention", sentence=None, translated_tokens=None):
    """

    :param attention_heads: attention matrix for each attn-head [H, Lq, Lkv]
    :param name:
    :param sentence:
    :param translated_tokens:
    :return:
    """
    # in_tokens = tf.convert_to_tensor([sentence])
    # in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    # in_tokens = tokenizers.pt.lookup(in_tokens)[0]
    # in_tokens

    fig = plt.figure(figsize=(12, 6))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h + 1)

        plot_attention_head(head)

        ax.set_xlabel(f'Head {h + 1}')

    plt.tight_layout()
    # plt.title(name)
    plt.show()


'''
 sample latent z based on actual test data x
 :param if_decode: reconstruct using the latent codes (generated by prior for example...)
'''


def generate_and_save_waves(model, epoch, test_sample, level=0, if_decode=False, latent_code=None, if_sample=False,
                            sample_mode=0, prior_model=None,
                            if_quantized=False, channel_last=False,
                            print_codebook_distribution=False,
                            sampler=None):
    # Codebook Vector usage (Put ahead reconstruction, given that vq preserve the last N_t for now)
    if print_codebook_distribution:
        vq_usage = model.get_quantizer().get_usage_count()
        print("Snapshot of Codebook Vector usage, Size(K/D, ): {}; SUM (last Batch size x latent_len){} ".format(
            tf.shape(vq_usage), tf.math.reduce_sum(vq_usage)))
        fig = plt.figure(figsize=(10, 5))
        plt.title("VQ Codebook Vector Usage Count")
        # plt.hist(vq_usage.numpy()) # should use bar...
        plt.bar(x=np.arange(len(vq_usage)), height=vq_usage.numpy())
        plt.show()

    print("[DEBUG] TOTAL Number of Trainable Weights for model: {}".format(
        sum([np.prod(var.numpy().shape) for var in model.vqvaes[level].trainable_variables])))

    # Direct Reconstruction (x -> x')
    input = test_sample[0] if isinstance(test_sample, tuple) else test_sample
    if channel_last:
        input = np.transpose(test_sample, [0, 2, 1])
    predictions = model.vqvaes[level](input, training=False).numpy()
    print("Reconstructed Output: ", predictions.shape)
    fig = plt.figure(figsize=(18, 12))
    # ret = []
    ret = predictions
    # if if_quantized:  # mu-law transformation
    #     # recover int waveform to float waveform
    #     # Greedy Sampling
    #     predictions = np.argmax(predictions, axis=-1)
    #     print("Sampled quantized: ", predictions.shape, predictions[0])
    #
    #     predictions = mu_law_decode(predictions, QUANTIZATION_CHANNELS)
    #
    #     print("Max: {}, Min: {}".format(np.amax(predictions), np.amin(predictions)))

    for i in range(predictions.shape[0]):
        waves = predictions[i]
        plt.subplot(4, 3, i + 1)

        if i == 9:
            print("X': ", waves.squeeze())
            print(f"X range: [{np.amax(waves)}, {np.amin(waves)}]")

        librosa.display.waveplot(waves.squeeze(), sr=SAMPLE_RATE)
        plt.title(f"WavePlot - {IDX_TO_GENRES[i]}")
        # plt.tight_layout()
        # ret.append(predictions[i])

    plt.show()

    if if_decode:
        # (N, T_l)
        assert latent_code is not None

        recons = model.decode(latent_code, level=level).numpy()

        print("-------------------------------- Reconstruction from latent codes... --------------------------")

        fig = plt.figure(figsize=(18, 12))

        for i in range(recons.shape[0]):
            waves = recons[i]
            plt.subplot(4, 3, i + 1)

            if i == 9:
                print("X': ", waves.squeeze())
                print(f"X range: [{np.amax(waves)}, {np.amin(waves)}]")

            librosa.display.waveplot(waves.squeeze(), sr=SAMPLE_RATE)
            plt.title(f"WavePlot - {IDX_TO_GENRES[i]}")
            # # plt.tight_layout()
            # ret.append(predictions[i])

        plt.show()

        ret = recons

    if if_sample:
        assert prior_model is not None or sampler is not None
        print(f"------------------------------- Auto-regressive Sampling in process..........-----------------------------")

        # sampled_codes, sampled_attn_w = prior_model.sample(n_samples=4)
        # TODO: Sample with conditioning on upper-level latent codes

        # sample_labels = [*IDX_TO_GENRES.keys()]

        assert isinstance(test_sample, tuple)
        sample_labels = test_sample[1]
        latent_codes_upper = None

        if sample_mode == 0:
            print(f"--------------[V0] Sampling with Ground-Truth Upper level decoded tokens")
            _, *latent_codes_cond = prior_model.vqvae.encode(test_sample[0], start_level=prior_model.level, end_level=prior_model.levels)
            latent_codes_upper = latent_codes_cond[0] if prior_model.level != prior_model.levels - 1 else None
            if latent_codes_upper is not None:
                print(f'[DEBUG] Shape of Upper Latent Codes: {latent_codes_upper.numpy().shape}')

            sampled_codes = prior_model.sample(n_samples=len(sample_labels), y=tf.constant(sample_labels),
                                               z_cond=latent_codes_upper)

        elif sample_mode == 1:
            print(f"--------------[V1] Single Level Sampling")
            sampled_codes = prior_model.sample(n_samples=len(sample_labels), y=tf.constant(sample_labels),
                                               z_cond=None)

        elif sample_mode == 2:
            print(f"--------------[V2] Full Levels Sampling")
            assert sampler is not None
            assert len(sampler.priors) == len(model.vqvaes)

            sampled_codes_all_levels = sampler.sample(n_samples=len(sample_labels), y_genre=tf.constant(sample_labels))
            sampled_codes = sampled_codes_all_levels[level]


        # sampled_codes = prior_model.sample(n_samples=len(sample_labels), y=tf.constant(sample_labels), z_cond=latent_codes_upper)

        print(f"Sampled output shape: {tf.shape(sampled_codes)}, with start token: {sampled_codes[:, 0]}")
        if np.sum(model.num_embeddings*np.ones_like(sampled_codes[:, 0].numpy()) == sampled_codes[:, 0].numpy()) == len(sampled_codes):
            print("Remove Start token...")
            sampled_codes = sampled_codes[:, 1:]

        print(sampled_codes, sampled_codes.shape)

        # # remove the start token
        # sampled_recons = model.decode(sampled_codes[:, 1:], level=level).numpy()
        sampled_recons = model.decode(sampled_codes, level=level).numpy()
        print(sampled_recons.shape)

        print("-------------------------------- Reconstruction from Prior Sampling (non-prime)... --------------------------")

        fig = plt.figure(figsize=(18, 12))

        for i in range(sampled_recons.shape[0]):
            waves = sampled_recons[i]
            plt.subplot(4, 3, i + 1)

            if i == 3:
                print("X': ", waves.squeeze())
                print(f"X range: [{np.amax(waves)}, {np.amin(waves)}]")

            librosa.display.waveplot(waves.squeeze(), sr=SAMPLE_RATE)
            plt.title(f"WavePlot - {IDX_TO_GENRES[sample_labels[i]]}")
            # # plt.tight_layout()
            # ret.append(predictions[i])

        plt.show()

        return sampled_recons, sampled_codes

    return ret


def decode_latent(model, sampled_codes, level):
    print(f"Sampled output shape: {tf.shape(sampled_codes)} (with Start Token...)")
    print(sampled_codes)

    # remove the start token
    sampled_recons = model.decode(sampled_codes, level=level).numpy()
    print(sampled_recons.shape)

    print(
        "-------------------------------- Reconstruction from Prior Sampling (non-prime)... --------------------------")

    fig = plt.figure(figsize=(18, 6))

    for i in range(sampled_recons.shape[0]):
        waves = sampled_recons[i]
        plt.subplot(1, 4, i + 1)

        if i == 3:
            print("X': ", waves.squeeze())
            print(f"X range: [{np.amax(waves)}, {np.amin(waves)}]")

        librosa.display.waveplot(waves.squeeze(), sr=SAMPLE_RATE)
        plt.title(f"WavePlot - {IDX_TO_GENRES[i]}")

    plt.show()

    return sampled_recons
