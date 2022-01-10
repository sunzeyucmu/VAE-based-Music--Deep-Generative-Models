from itertools import chain
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from encdec import Encoder, Decoder, print_dec_layer
from VectorQuantizer import VectorQuantizer
from data_utils import STFT_ARGS, spectral, norm

"""
 Single Level VQ-VAE
"""


def get_vqvae(input_shape, encoder, decoder, vq, level=0):
    inputs = keras.Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents, _ = vq(encoder_outputs)
    # print("VQ output:", quantized_latents)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae_{}".format(level))


class VQVAE(keras.models.Model):
    """
      @latent_dim
      @levels: number of stacked VQ-VAE (could be hierarchical like VQ-VAE-2, or independent like jukebox paper proposed)
    """

    def __init__(self,
                 input_shape,
                 levels,
                 latent_dim, down_depth, strides, num_embeddings=128, residual_width=64,
                 residual_depth=4, dilation_factor=1, train_variance=1.0, **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.levels = levels
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        # self.depth = depth

        # self.vqvae = get_vqvae(input_shape, self.latent_dim, self.depth, down_depth, strides, self.num_embeddings, residual_width, residual_depth, dilation_factor)
        # self.vqvae = VQVAE(input_shape, self.latent_dim, self.depth, down_depth, strides, self.num_embeddings, residual_width, residual_depth, dilation_factor)

        # self.vq = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        # Bottlenecks: separate for each level
        self.vqs = [VectorQuantizer(num_embeddings, latent_dim, level=level, name="vector_quantizer_{}".format(level)) for level in
                    range(levels)]
        # self.encoder = Encoder(output_dim=latent_dim, residual_width=residual_width,
        #                        residual_depth=residual_depth,
        #                        depth=depth, down_depth=down_depth, strides=strides,
        #                        dilation_factor=dilation_factor)
        # Encoders: from bottom to top; with hop_length increasing (8, 32, 128)
        self.encoders = [Encoder(output_dim=latent_dim, residual_width=residual_width,
                                 residual_depth=residual_depth,
                                 depth=level + 1, down_depth=down_depth[:level + 1], strides=strides[:level + 1],
                                 dilation_factor=dilation_factor, name="encoder_{}".format(level)) for level in
                         range(levels)]
        # self.decoder = Decoder(output_dim=input_shape[-1], embed_width=latent_dim, residual_width=residual_width,
        #                        residual_depth=residual_depth,
        #                        depth=depth, down_depth=down_depth, strides=strides,
        #                        dilation_factor=dilation_factor)
        # Decoders: from bottom to top;
        self.decoders = [Decoder(output_dim=input_shape[-1], embed_width=latent_dim, residual_width=residual_width,
                                 residual_depth=residual_depth,
                                 depth=level + 1, down_depth=down_depth[:level + 1], strides=strides[:level + 1],
                                 dilation_factor=dilation_factor, name="decoder_{}".format(level)) for level in
                         range(levels)]

        # self.vqvae = get_vqvae(input_shape, self.encoder, self.decoder, self.vq)
        # VQVAES: from bottom to top
        self.vqvaes = [get_vqvae(input_shape, self.encoders[level], self.decoders[level], self.vqs[level], level)
                       for level in range(levels)]

        # Metrics Trackers
        ## Overall
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.spectral_loss_tracker = keras.metrics.Mean(name='spectral_loss')
        ## By Level
        self.level_loss_trackers = [keras.metrics.Mean(name="[{}]level_loss".format(level)) for level in range(levels)]
        self.recon_loss_trackers = [keras.metrics.Mean(name="[{}]recon_loss".format(level)) for level in range(levels)]
        self.vq_loss_trackers = [keras.metrics.Mean(name="[{}]vq_loss".format(level)) for level in range(levels)]
        self.spectral_loss_trackers = [keras.metrics.Mean(name="[{}]spectral_loss".format(level)) for level in range(levels)]
        # Loss Func for reconstruction
        self.loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.spectral_loss_tracker,
            *self.level_loss_trackers,
            *self.recon_loss_trackers,
            *self.vq_loss_trackers,
            *self.spectral_loss_trackers,
        ]

    """
       x being tuple of (batched_audio_waveform, genre_label)
    """

    # @tf.function
    def train_step(self, data):
        # print("Step Training....")
        x = data[0]
        commit_losses = []
        recon_losses = []
        spectral_losses = []
        level_losses = []
        total_loss = tf.zeros(())
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            for level in range(self.levels): # bottom to top
                reconstructions = self.vqvaes[level](x)

                # Calculate the losses.
                reconstruction_loss = tf.reduce_mean(self.loss_fn(x, reconstructions))
                # Spectral Loss
                spectral_loss = tf.reduce_mean(self._multispectral_loss(x, reconstructions))
                # commitment loss
                commit_loss = sum(self.vqvaes[level].losses)

                level_loss = reconstruction_loss + commit_loss + spectral_loss
                commit_losses.append(commit_loss)
                recon_losses.append(reconstruction_loss)
                spectral_losses.append(spectral_loss)
                level_losses.append(level_loss)

                total_loss += level_loss

        # Backpropagation.
        # grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        trainable_vars = list(chain.from_iterable([m.trainable_variables for m in self.vqvaes]))
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return self.update_metrics(level_losses, recon_losses, commit_losses, spectral_losses)

    def test_step(self, data):
        x = data[0]

        commit_losses = []
        recon_losses = []
        spectral_losses = []
        level_losses = []
        # Outputs from the VQ-VAE.
        for level in range(self.levels):  # bottom to top
            reconstructions = self.vqvaes[level](x)

            # Calculate the losses.
            reconstruction_loss = tf.reduce_mean(self.loss_fn(x, reconstructions))
            # Spectral Loss
            spectral_loss = tf.reduce_mean(self._multispectral_loss(x, reconstructions))
            # commitment loss
            commit_loss = sum(self.vqvaes[level].losses)

            level_loss = reconstruction_loss + commit_loss + spectral_loss
            commit_losses.append(commit_loss)
            recon_losses.append(reconstruction_loss)
            spectral_losses.append(spectral_loss)
            level_losses.append(level_loss)

        return self.update_metrics(level_losses, recon_losses, commit_losses, spectral_losses)

    '''
      For CallBack model call
    '''

    def call(self, x):
        # take the bottom level result... for now...
        # return self.vqvae(x)
        return self.vqvaes[0](x)

    def update_metrics(self, level_losses, recon_losses, commit_losses, spectral_losses):
        # Loss tracking.
        self.total_loss_tracker.update_state(sum(level_losses))
        self.reconstruction_loss_tracker.update_state(sum(recon_losses))
        self.vq_loss_tracker.update_state(sum(commit_losses))
        self.spectral_loss_tracker.update_state(sum(spectral_losses))

        level_metrics = []

        for level in range(self.levels):
            self.level_loss_trackers[level].update_state(level_losses[level])
            self.recon_loss_trackers[level].update_state(recon_losses[level])
            self.vq_loss_trackers[level].update_state(commit_losses[level])
            self.spectral_loss_trackers[level].update_state(spectral_losses[level])
            metrics = {self.level_loss_trackers[level].name: self.level_loss_trackers[level].result(),
                       self.recon_loss_trackers[level].name: self.recon_loss_trackers[level].result(),
                       self.vq_loss_trackers[level].name: self.vq_loss_trackers[level].result(),
                       self.spectral_loss_trackers[level].name: self.spectral_loss_trackers[level].result()}
            metrics.update({m.name: m.result() for m in self.vqs[level].metrics})
            level_metrics.append(metrics)

        # Log results.
        ret_metrics = dict(loss=self.total_loss_tracker.result(),
                           recon_loss=self.reconstruction_loss_tracker.result(),
                           vqvae_loss=self.vq_loss_tracker.result(),
                           spectral_loss=self.spectral_loss_tracker.result())
        ## Another way to organize/concate the metrics...
        # ## VQ metrics
        # vq_metrics = {m.name: m.result() for vq in self.vqs for m in vq.metrics}
        #
        # ## Encoder metrics: TODO
        #
        # ret_metrics.update(dict(
        #     enc_place_holder=0.0,
        #     **vq_metrics,
        #     **{m.name: m.result() for m in self.level_loss_trackers},
        #     **{m.name: m.result() for m in self.recon_loss_trackers},
        #     **{m.name: m.result() for m in self.vq_loss_trackers},
        #     **{m.name: m.result() for m in self.spectral_loss_trackers}
        # ))
        ret_metrics.update({k: v for d in level_metrics for k, v in d.items()})

        return ret_metrics

    def get_quantizer(self):
        return self.vqs[0]

    def _multispectral_loss(self, target, recon, **kwargs):
        losses = []
        # shape = tf.shape(target)
        # target_ = tf.reshape(target, [shape[0]])

        for n_fft, hop_len, window_size in zip(*STFT_ARGS):
            spec_tar = spectral(tf.squeeze(target), n_fft, hop_len, window_size)
            spec_recon = spectral(tf.squeeze(recon), n_fft, hop_len, window_size)
            # 1. complete spectral loss
            # losses.append(norm(spec_tar - spec_recon))
            # scale by bandwidth
            losses.append(norm(spec_tar - spec_recon) / norm(spec_tar))

        # return sum(losses) / len(losses)
        # print("[DEBUG] Spectral Losses: ", losses)

        losses = tf.stack(losses, axis=-1)  # (N, S)
        return tf.reduce_mean(losses, axis=-1)


if __name__ == '__main__':
    print('VQ-VAE module')
    sample_batch = tf.random.uniform([32, 28160, 1])
    sample_y = tf.random.uniform([32, ])

    # vqvae_trainer = VQVAE(sample_batch.shape[1:], latent_dim=64, num_embeddings=512, depth=2, down_depth=[5, 3], strides=[2, 2], dilation_factor=3)  # codebook size
    # vqvae_trainer = VQVAE(sample_batch.shape[1:], latent_dim=64, num_embeddings=512, depth=1, down_depth=[5],
    #                       strides=[2], dilation_factor=3)
    # vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    #
    # vqvae_trainer.vqvae.summary()
    #
    # vqvae_trainer.encoder.model.summary()
    #
    # vqvae_trainer.decoder.model.summary()
    #
    # # print_dec_layer(vqvae_trainer.decoder)
    #
    # # Validate train_step
    # vqvae_trainer.fit(x=sample_batch, y=sample_y, batch_size=8, epochs=4)

    # Stack of VQ-VAE
    # vqvae = VQVAE(sample_batch.shape[1:], levels=1, latent_dim=64, num_embeddings=512, down_depth=[5], strides=[2], dilation_factor=3)
    vqvae = VQVAE(sample_batch.shape[1:], levels=2, latent_dim=64, num_embeddings=512, down_depth=[5, 3], strides=[2, 2], dilation_factor=3, residual_width=32)

    for l, model, enc, dec, vq in zip(range(vqvae.levels), vqvae.vqvaes, vqvae.encoders, vqvae.decoders, vqvae.vqs):
        print("======================VQ-VAE: {}============================".format(l))
        model.summary()
        enc.model.summary()
        dec.model.summary()
        # vq.model.summary()

    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.fit(x=sample_batch, y=sample_y, batch_size=8, epochs=4)

