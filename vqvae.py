import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from encdec import Encoder, Decoder, print_dec_layer
from VectorQuantizer import VectorQuantizer

"""
 Single Level VQ-VAE
"""
def get_vqvae(input_shape, encoder, decoder, vq):
    inputs = keras.Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents, _= vq(encoder_outputs)
    # print("VQ output:", quantized_latents)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class VQVAE(keras.models.Model):
    """
      @latent_dim
      @
    """

    def __init__(self, input_shape, latent_dim, depth, down_depth, strides, num_embeddings=128, residual_width=64,
                 residual_depth=4, dilation_factor=1, train_variance=1.0, **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.depth = depth

        # self.vqvae = get_vqvae(input_shape, self.latent_dim, self.depth, down_depth, strides, self.num_embeddings, residual_width, residual_depth, dilation_factor)
        # self.vqvae = VQVAE(input_shape, self.latent_dim, self.depth, down_depth, strides, self.num_embeddings, residual_width, residual_depth, dilation_factor)

        self.vq = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        self.encoder = Encoder(output_dim=latent_dim, residual_width=residual_width,
                               residual_depth=residual_depth,
                               depth=depth, down_depth=down_depth, strides=strides,
                               dilation_factor=dilation_factor)
        self.decoder = Decoder(output_dim=input_shape[-1], embed_width=latent_dim, residual_width=residual_width,
                               residual_depth=residual_depth,
                               depth=depth, down_depth=down_depth, strides=strides,
                               dilation_factor=dilation_factor)

        self.vqvae = get_vqvae(input_shape, self.encoder, self.decoder, self.vq)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    """
       x being tuple of (batched_audio_waveform, genre_label)
    """

    # @tf.function
    def train_step(self, data):
        x = data[0]
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            # reconstruction_loss = (
            #     tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            # )
            reconstruction_loss = tf.reduce_mean(self.loss_fn(x, reconstructions))

            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        ret_metrics = dict(loss=self.total_loss_tracker.result(),
                           reconstruction_loss=self.reconstruction_loss_tracker.result(),
                           vqvae_loss=self.vq_loss_tracker.result())
        ## VQ metrics
        vq_metrics = {m.name: m.result() for m in self.vq.metrics}

        ## Encoder metrics: TODO

        ret_metrics.update(dict(
            enc_place_holder=0.0,
            **vq_metrics
        ))

        return ret_metrics

    '''
      For CallBack model call
    '''

    def call(self, x):
        return self.vqvae(x)

    def get_quantizer(self):
        return self.vq


if __name__ == '__main__':
    print('VQ-VAE module')
    sample_batch = tf.random.uniform([32, 28000, 1])
    
    vqvae_trainer = VQVAE(sample_batch.shape[1:], latent_dim=64, num_embeddings=512, depth=2, down_depth=[5, 3], strides=[2, 2], dilation_factor=3)  # codebook size
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

    vqvae_trainer.vqvae.summary()

    vqvae_trainer.encoder.model.summary()

    vqvae_trainer.decoder.model.summary()

    print_dec_layer(vqvae_trainer.decoder)

