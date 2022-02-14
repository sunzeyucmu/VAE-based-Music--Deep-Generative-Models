import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.autoregressive.autoregressive_fmha import FMHABasedAutoregressiveModel
from autoregressive import loss_function, accuracy_function
from utils import tf_utils


class PriorMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_dataset, train_samples, test_samples, val_interval=10, sample_interval=50, **kwargs):
        super(PriorMonitor, self).__init__(**kwargs)
        self.val_interval = val_interval
        self.sample_interval = sample_interval
        self.val_dataset = test_dataset
        self.train_samples = train_samples
        self.test_samples = test_samples

    def on_epoch_end(self, epoch, logs=None):

        # Periodic Inspect
        if epoch % self.val_interval == 0:
            # Reset the metrics!!!
            print("\nResetting the metrics...")
            for m in self.model.metrics:
                m.reset_state()

            print("\n[DEBUG] This is Callback Monitor: End of Epoch", epoch)
            print("---------------------------Running Validation DataSet---------------------------")
            self.model.evaluate(self.val_dataset)

            print(f"-------------------------------------Validate Test Samples Performance------------------------------------------")

            pred_logits_test, target_test, attn_weights_test, loss_test, accuracy_test = self.model(self.test_samples)

            print(f'Testing Samples Loss {loss_test:.4f}; Perplexity (exp of loss_per_word): {tf.math.exp(loss_test):.5f}; Accuracy {accuracy_test:.4f}')
            print(">>>>>>>>>>> Top 100 of Test Target: ", target_test[0][:100])
            print(">>>>>>>>>>> Top 100 of Test Preds: ", tf.argmax(pred_logits_test, axis=2)[0][:100])

            for k, v in attn_weights_test.items():
                print(k)
                print(v.shape)
                tf_utils.plot_attention_weights(v[0])

            tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.test_samples, level=self.model.level, if_decode=True,
                                             latent_code=tf.argmax(pred_logits_test, axis=-1), if_sample=False)

            # Training Samples
            print(f"-------------------------------------Validate Training Samples Performance--------------------------------------")

            pred_logits, target, attn_weights, loss, accuracy = self.model(self.train_samples)

            print(
                f'Training Samples Loss {loss:.4f}; Perplexity (exp of loss_per_word): {tf.math.exp(loss):.5f}; Accuracy {accuracy:.4f}')
            print(">>>>>>>>>>> Top 100 of Train Target: ", target[0][:100])
            print(">>>>>>>>>>> Top 100 of Train Preds: ", tf.argmax(pred_logits, axis=2)[0][:100])

            # Plot Attentions
            for k, v in attn_weights.items():
                print(k)
                print(v.shape)
                tf_utils.plot_attention_weights(v[0])

            # tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level, if_decode=True,
            #                                  latent_code=tf.argmax(pred_logits, axis=-1), if_sample=False)

            # Greedy Sampling also
            if epoch % self.sample_interval == 0:
                # if epoch+1 % 100 ==0:
                # Sampling is a bit costy....
                tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level, if_decode=True,
                                                 latent_code=tf.argmax(pred_logits, axis=-1), if_sample=True,
                                                 prior_model=self.model.prior)
            else:
                tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level, if_decode=True,
                                                 latent_code=tf.argmax(pred_logits, axis=-1), if_sample=False)

            return pred_logits, attn_weights


class Prior(keras.Model):
    def __init__(self,
                 level,
                 z_shapes,
                 bins,  # codebook size
                 down_depth,
                 strides,
                 vqvae_model,
                 prior_kwargs,
                 x_cond_kwargs,
                 **kwargs):
        """

        :param level:
        :param z_shapes: shape of the latent inputs at each vq-vae level
        :param kwargs:
        """
        super(Prior, self).__init__(**kwargs)
        assert len(down_depth) == len(strides) == len(z_shapes)
        # assert vqvae_model.num_embeddings == bins

        self.level = level
        self.z_shapes = z_shapes
        self.levels = len(z_shapes)
        self.z_shape = z_shapes[level]
        self.bins = bins

        # Shape of Training Audio Batch
        self.sample_shape = tf.TensorShape([np.cumprod(z_shapes[0])[0] * (strides[0] ** down_depth[0]), 1])
        # TODO: cross validate with VQVAE model
        self.train_step_signature = [
            tf.TensorSpec(shape=(None, *self.sample_shape), dtype=tf.float32),
        ]

        # TO DO
        self.vqvae = vqvae_model
        self.prior = FMHABasedAutoregressiveModel(context_length=self.z_shape,
                                                  target_vocab_size=self.bins,
                                                  width=prior_kwargs['width'],
                                                  depth=prior_kwargs['depth'],
                                                  heads=prior_kwargs['heads'],
                                                  blocks=prior_kwargs['blocks'],
                                                  attn_stacks=prior_kwargs['attn_stacks'],
                                                  drop_out_rate=prior_kwargs['drop_out_rate'],
                                                  zq_shapes=self.z_shapes,  # [(16,), (4,)],
                                                  level=self.level,
                                                  levels=self.levels,
                                                  downs=down_depth,
                                                  strides=strides,
                                                  cond_kwargs=x_cond_kwargs)

        self.ent_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # Metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy_tracker = tf.keras.metrics.Mean(name='train_accuracy')
        ## TODO: perplexity

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.train_accuracy_tracker,
        ]

    def encode(self, x, start_level, end_level):
        return NotImplementedError

    def decode(self, zs, start_level, end_level):
        return NotImplementedError

    def call(self, x):
        """
        Forward Pass of raw audio file to prior model
        :param x: [N, T, 1/C]
        :return:
        """
        latent_zs_train = self.vqvae.encode(x, start_level=self.level, end_level=self.levels)

        latent_input = tf.pad(latent_zs_train[self.level][:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        # print(latent_input.numpy())
        target = latent_zs_train[self.level]

        # pred_logits, attn_weights = model(latent_input, training=False)
        pred_logits, attn_weights = self.prior(latent_input, training=False, x_cond=latent_zs_train[self.level + 1])

        loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token
        acc = accuracy_function(target, pred_logits)

        return pred_logits, target, attn_weights, loss, acc


    @tf.function
    def train_step(self, x):
        """
        :param x: tuple of (N, T, 1) the raw audio waveform!
        :return:
        """
        inputs = x[0]
        latent_zs = self.vqvae.encode(inputs, start_level=self.level, end_level=self.levels)
        latent_codes = latent_zs[self.level]
        latent_codes_upper = latent_zs[self.level + 1] if self.level != self.levels - 1 else None

        latent_input = tf.pad(latent_codes[:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        # print(latent_input.numpy(), latent_codes.shape)
        target = latent_codes

        with tf.GradientTape() as tape:
            pred_logits, attn_weights = self.prior(latent_input, x_cond=latent_codes_upper, training=True)

            loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token

        gradients = tape.gradient(loss, self.prior.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.prior.trainable_variables))

        self.train_loss_tracker(loss)
        self.train_accuracy_tracker(accuracy_function(target, pred_logits))

        # return attn_weights
        return {
            "loss": self.train_loss_tracker.result(),
            "perplexity(per word)": tf.math.exp(self.train_loss_tracker.result()),
            "accuracy": self.train_accuracy_tracker.result(),
        }

    def test_step(self, x):
        X = x[0]
        latent_zs = self.vqvae.encode(X, start_level=self.level, end_level=self.levels)

        latent_input = tf.pad(latent_zs[self.level][:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        latent_codes_upper = latent_zs[self.level + 1] if self.level != self.levels - 1 else None

        target = latent_zs[self.level]

        pred_logits, attn_weights = self.prior(latent_input, training=False, x_cond=latent_codes_upper)

        loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token
        self.train_loss_tracker(loss)
        self.train_accuracy_tracker(accuracy_function(target, pred_logits))

        return {
            "loss": self.train_loss_tracker.result(),
            "perplexity(per word)": tf.math.exp(self.train_loss_tracker.result()),
            "accuracy": self.train_accuracy_tracker.result(),
        }

    def sample_level(self, zs, level):
        """
        Sample a single window of length=n_ctx at level=level

        :param level:
        :param zs:
        :return:
        """
        return NotImplementedError


if __name__ == '__main__':
    print('Prior module')

    prior_kwargs = dict(width=128, depth=6, heads=2, blocks=4, attn_stacks=1)
    x_cond_kwargs = dict(dilation_factor=3, dilation_cycle=4, residual_width=32, residual_depth=8)
    prior = Prior(0, [(256,), (64,), (16,)], bins=512, down_depth=[3, 2, 2], strides=[2, 2, 2],
                  prior_kwargs=prior_kwargs, x_cond_kwargs=x_cond_kwargs, vqvae_model=None)

    print(prior.sample_shape, prior.train_step_signature)
