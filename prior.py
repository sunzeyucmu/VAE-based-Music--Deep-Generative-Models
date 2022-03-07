import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from src.conditioner.label_conditioners import LabelConditioner
from src.autoregressive.autoregressive_fmha import FMHABasedAutoregressiveModel
from autoregressive import loss_function, accuracy_function
from vqvae import VQVAE
from utils import tf_utils


# class PriorMonitor(tf.keras.callbacks.Callback):
#     """A callback to generate and save images after each epoch"""
#
#     def __init__(self, test_dataset, train_samples, test_samples, ckpt_manager, val_interval=10, sample_interval=50,
#                  ckpt_interval=20, **kwargs):
#         super(PriorMonitor, self).__init__(**kwargs)
#         self.val_interval = val_interval
#         self.sample_interval = sample_interval
#         self.val_dataset = test_dataset
#         self.train_samples = train_samples
#         self.test_samples = test_samples
#         self.ckpt_manager = ckpt_manager
#         self.ckpt_interval = ckpt_interval
#
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % self.ckpt_interval == 0:
#             ckpt_save_path = self.ckpt_manager.save()
#             print(f'\nSaving checkpoint for epoch {epoch + 1} at {ckpt_save_path}\n')
#
#         # Periodic Inspect
#         if epoch % self.val_interval == 0:
#             # Reset the metrics!!!
#             print("\nResetting the metrics...")
#             for m in self.model.metrics:
#                 m.reset_state()
#
#             print("\n[DEBUG] This is Callback Monitor: End of Epoch", epoch)
#             print("---------------------------Running Validation DataSet---------------------------")
#             self.model.evaluate(self.val_dataset)
#
#             print(
#                 f"-------------------------------------Validate Test Samples Performance------------------------------------------")
#             if isinstance(self.test_samples, tuple):
#                 print(f'---------Labels provided for Test Samples: {self.test_samples[1]}')
#
#             pred_logits_test, target_test, attn_weights_test, loss_test, accuracy_test = self.model(self.test_samples)
#
#             print(
#                 f'Testing Samples Loss {loss_test:.4f}; Perplexity (exp of loss_per_word): {tf.math.exp(loss_test):.5f}; Accuracy {accuracy_test:.4f}')
#             print(">>>>>>>>>>> Top 100 of Test Target: ", target_test[0][:100])
#             print(">>>>>>>>>>> Top 100 of Test Preds: ", tf.argmax(pred_logits_test, axis=2)[0][:100])
#
#             for k, v in attn_weights_test.items():
#                 print(k)
#                 print(v.shape)
#                 tf_utils.plot_attention_weights(v[0])
#
#             tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.test_samples, level=self.model.level,
#                                              if_decode=True,
#                                              latent_code=tf.argmax(pred_logits_test, axis=-1), if_sample=False)
#
#             # Training Samples
#             print(
#                 f"-------------------------------------Validate Training Samples Performance--------------------------------------")
#             if isinstance(self.train_samples, tuple):
#                 print(f'---------Labels provided for Test Samples: {self.train_samples[1]}')
#
#             pred_logits, target, attn_weights, loss, accuracy = self.model(self.train_samples)
#
#             print(
#                 f'Training Samples Loss {loss:.4f}; Perplexity (exp of loss_per_word): {tf.math.exp(loss):.5f}; Accuracy {accuracy:.4f}')
#             print(">>>>>>>>>>> Top 100 of Train Target: ", target[0][:100])
#             print(">>>>>>>>>>> Top 100 of Train Preds: ", tf.argmax(pred_logits, axis=2)[0][:100])
#
#             # Plot Attentions
#             for k, v in attn_weights.items():
#                 print(k)
#                 print(v.shape)
#                 tf_utils.plot_attention_weights(v[0])
#
#             # tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level, if_decode=True,
#             #                                  latent_code=tf.argmax(pred_logits, axis=-1), if_sample=False)
#
#             # Greedy Sampling also
#             if epoch % self.sample_interval == 0:
#                 # if epoch+1 % 100 ==0:
#                 # Sampling is a bit costy....
#                 tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level,
#                                                  if_decode=True,
#                                                  latent_code=tf.argmax(pred_logits, axis=-1), if_sample=True,
#                                                  prior_model=self.model)
#             else:
#                 tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level,
#                                                  if_decode=True,
#                                                  latent_code=tf.argmax(pred_logits, axis=-1), if_sample=False)
#
#             return pred_logits, attn_weights


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
                 prior_monitor=None,
                 genre_classes=None,
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
        self.context_length = tf.reduce_prod(self.z_shape)  # (T, 1) -> T
        self.bins = bins
        self.genre_bins = genre_classes

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

        # Label Conditioning (prepended to head of sequence embeddings)
        if self.genre_bins is not None:
            print(f"[DEBUG] Initialize Label Conditioner with Genre Classes: {self.genre_bins}")
            self.label_conditioner = LabelConditioner(genre_bins=self.genre_bins, width=prior_kwargs['width'])

        self.ent_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # Metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy_tracker = tf.keras.metrics.Mean(name='train_accuracy')
        ## TODO: perplexity

        # CallBack Monitor... TODO: Move this to a fully customized trainer function
        self.train_monitor = prior_monitor
        # self.iters = 0

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

    def get_cond(self, zs, start, end):
        """
        Retrieve upper-level latent codes (already sampled) matching loc of [start, end) of current level
        :param zs:
        :param start:
        :param end:
        :return:
        """
        # TODO: move the upper level conditioning to Prior module
        return self.prior.get_cond(zs, start, end)

    def set_train_monitor(self, train_monitor):
        self.train_monitor = train_monitor

    def call(self, inputs):
        """
        Forward Pass of raw audio file to prior model
        :param inputs: [N, T, 1/C]
        :return:
        """
        if isinstance(inputs, tuple):
            x, y = inputs
            print(f"[DEBUG] Label present: {tf.shape(y)}")
        else:
            print(f"[DEBUG] Label not provided")
            x = inputs
            y = None

        # latent_zs_train = self.vqvae.encode(x, start_level=self.level, end_level=self.levels)
        #
        # latent_input = tf.pad(latent_zs_train[self.level][:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
        #                       constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        # # print(latent_input.numpy())
        # target = latent_zs_train[self.level]
        latent_codes, *latent_codes_cond = self.vqvae.encode(x, start_level=self.level, end_level=self.levels)
        # take only the upper level tokens
        latent_codes_upper = latent_codes_cond[0] if self.level != self.levels - 1 else None

        latent_input = tf.pad(latent_codes[:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        target = latent_codes

        y_cond = None
        if y is not None:
            assert self.label_conditioner is not None
            y_cond = self.label_conditioner(y, training=False)

        pred_logits, attn_weights = self.prior(latent_input, training=False, x_cond=latent_codes_upper, y_cond=y_cond)

        loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token
        acc = accuracy_function(target, pred_logits)

        return pred_logits, target, attn_weights, loss, acc

    @tf.function
    def train_step(self, x, teacher_force_rate=0.2):
        """
        :param teacher_force_rate:
        :param x: tuple of (N, T, 1) the raw audio waveform!
        :return:
        """
        ## DEBUG
        print(f"Input Len: {len(x)}, Content: {x}")
        # inputs = x[0]

        if isinstance(x, tuple):
            inputs, y = x
        else:
            y = None
        # latent_zs = self.vqvae.encode(inputs, start_level=self.level, end_level=self.levels)
        # latent_codes = latent_zs[self.level]
        # latent_codes_upper = latent_zs[self.level + 1] if self.level != self.levels - 1 else None

        latent_codes, *latent_codes_cond = self.vqvae.encode(inputs, start_level=self.level, end_level=self.levels)
        latent_codes_upper = latent_codes_cond[0] if self.level != self.levels - 1 else None

        latent_input = tf.pad(latent_codes[:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        # print(latent_input.numpy(), latent_codes.shape)
        target = latent_codes

        # y_cond = None
        # if y is not None:
        #     assert self.label_conditioner is not None
        #     y_cond = self.label_conditioner(y, training=True)

        with tf.GradientTape() as tape:
            y_cond = None
            if y is not None:
                assert self.label_conditioner is not None
                y_cond = self.label_conditioner(y, training=True)

            # Teacher Forcing
            forward_logits, _ = self.prior(latent_input, x_cond=latent_codes_upper, training=True,
                                           y_cond=y_cond)

            # TODO: Take Sampled Value?
            pred_latent = tf.argmax(forward_logits, axis=2)  # (N, T)
            pred_latent = tf.pad(pred_latent[:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                                 constant_values=self.bins - 1)
            print(f"[DEBUG] Apply Teacher Forcing with tf_rate: {teacher_force_rate}")
            idx = tf.random.uniform(tf.shape(pred_latent), minval=0, maxval=1, dtype=tf.float32) < teacher_force_rate

            batch_input = tf.where(idx, pred_latent,
                                   latent_input)  # (take model output as input for next step randomly)
            pred_logits, attn_weights = self.prior(batch_input, x_cond=latent_codes_upper, training=True,
                                                   y_cond=y_cond)

            # pred_logits, attn_weights = self.prior(latent_input, x_cond=latent_codes_upper, training=True,
            #                                        y_cond=y_cond)

            loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token

        variables = self.prior.trainable_variables if y_cond is None else self.prior.trainable_variables + self.label_conditioner.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_loss_tracker(loss)
        self.train_accuracy_tracker(accuracy_function(target, pred_logits))

        """
        Logging to Tensorboard, note this is EXPENSIVE!!!
        """
        if self.train_monitor is not None:
            with self.train_monitor.summary_writer.as_default():
                # Log the weights and gradients
                for var, grad in zip(variables, gradients):
                    tf.summary.histogram(name=f"[grad]{var.name}", data=grad,
                                         step=tf.summary.experimental.get_step())  # , step=self.iters) #step=self.train_monitor.step)
                    tf.summary.histogram(name=var.name, data=var,
                                         step=tf.summary.experimental.get_step())  # , step=self.iters) #step=self.train_monitor.step)

                # Log Only the Transformer's weights
                # for var in self.prior.transformer.trainable_variables:
                #     tf.summary.histogram(name=var.name, data=var, step=tf.summary.experimental.get_step()) #, step=self.iters) #step=self.train_monitor.step)

        # # Log the weights and gradients
        # for var, grad in zip(variables, gradients):
        #     tf.summary.histogram(name=f"[grad]{var.name}", data=grad)
        #
        # # Log Only the Transformer's weights
        # for var in self.prior.transformer.trainable_variables:
        #     tf.summary.histogram(name=var.name, data=var)

        # return attn_weights
        return {
            "loss": self.train_loss_tracker.result(),
            "perplexity(per word)": tf.math.exp(self.train_loss_tracker.result()),
            "accuracy": self.train_accuracy_tracker.result(),
        }

    def test_step(self, x):
        # X = x[0]
        if isinstance(x, tuple):
            X, y = x
        else:
            y = None
        # latent_zs = self.vqvae.encode(X, start_level=self.level, end_level=self.levels)
        #
        # latent_input = tf.pad(latent_zs[self.level][:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
        #                       constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        # latent_codes_upper = latent_zs[self.level + 1] if self.level != self.levels - 1 else None
        #
        # target = latent_zs[self.level]
        latent_codes, *latent_codes_cond = self.vqvae.encode(X, start_level=self.level, end_level=self.levels)
        latent_codes_upper = latent_codes_cond[0] if self.level != self.levels - 1 else None

        latent_input = tf.pad(latent_codes[:, :-1], paddings=[[0, 0], [1, 0]], mode='CONSTANT',
                              constant_values=self.bins - 1)  # TODO: this is temporary, using the label token embeddings instead!
        target = latent_codes

        y_cond = None
        if y is not None:
            assert self.label_conditioner is not None
            y_cond = self.label_conditioner(y, training=False)

        pred_logits, attn_weights = self.prior(latent_input, training=False, x_cond=latent_codes_upper, y_cond=y_cond)

        loss = loss_function(target, pred_logits, loss_fn=self.ent_loss_fn)  # loss per token
        self.train_loss_tracker(loss)
        self.train_accuracy_tracker(accuracy_function(target, pred_logits))

        return {
            "loss": self.train_loss_tracker.result(),
            "perplexity(per word)": tf.math.exp(self.train_loss_tracker.result()),
            "accuracy": self.train_accuracy_tracker.result(),
        }

    def sample(self, n_samples, z_cond=None, y=None, return_attn_weights=False):
        """
        Sample a single window of length=n_ctx at level=level

        :param z_cond:
        :param y:
        :param n_samples:
        :param level:
        :param zs:
        :return:
        """

        if z_cond is not None:
            tf.debugging.assert_equal(
                tf_utils.shape_list(z_cond)[0], n_samples,
                message=f"Batch Size not matching, Expected:{n_samples}, Getting: {tf_utils.shape_list(z_cond)[0]}",
                summarize=None, name=None
            )
            # TODO: move z_cond up_sampling to this prior layer

        y_cond = None
        if y is not None:
            tf.debugging.assert_equal(
                tf_utils.shape_list(y)[0], n_samples,
                message=f"Batch Size not matching, Expected:{n_samples}, Getting: {tf_utils.shape_list(y)[0]}",
                summarize=None, name=None
            )
            print(f"[DEBUG] Labels to Initiate the Sampling: {y}")

            # Generate the Label Embedding from y
            assert self.label_conditioner is not None
            y_cond = self.label_conditioner(y, training=False)

        return self.prior.sample(n_samples=n_samples, x_cond=z_cond, y_cond=y_cond,
                                 return_attention_weights=return_attn_weights)


if __name__ == '__main__':
    print('Prior module')

    prior_kwargs = dict(width=128, depth=6, heads=2, blocks=4, attn_stacks=1, drop_out_rate=0.1)
    x_cond_kwargs = dict(dilation_factor=3, dilation_cycle=4, residual_width=32, residual_depth=8)

    # inputs_batch = tf.random.normal([4, 2048, 1])
    inputs_batch = (tf.random.normal([4, 2048, 1]), tf.random.uniform((4,), dtype=tf.int64, minval=0, maxval=9))

    vqvae = VQVAE(inputs_batch[0].shape[1:], levels=3, latent_dim=64, num_embeddings=512, down_depth=[3, 2, 2],
                  strides=[2, 2, 2], dilation_factor=3, residual_width=32)
    prior = Prior(0, [(256,), (64,), (16,)], bins=512, down_depth=[3, 2, 2], strides=[2, 2, 2],
                  prior_kwargs=prior_kwargs, x_cond_kwargs=x_cond_kwargs, vqvae_model=vqvae, genre_classes=10)

    print(prior.sample_shape, prior.train_step_signature)

    print(f"Test Forward Pass")
    batch_out = prior(inputs_batch)
    init_label_emb = prior.label_conditioner.get_weights()[0]
    prior(inputs_batch)
    updated_label_emb = prior.label_conditioner.get_weights()[0]
    print(f"[DEBUG] Validate Label Conditioner Embeddings Update: {np.sum(updated_label_emb != init_label_emb)}")

    print(f"Test Backward Pass.....")
    init_label_emb = prior.label_conditioner.get_weights()[0]
    prior.compile(optimizer=tf.keras.optimizers.Adam())
    step_ret = prior.train_step(inputs_batch)
    updated_label_emb = prior.label_conditioner.get_weights()[0]
    print(f"[DEBUG] Validate Label Conditioner Embeddings Update: {np.sum(updated_label_emb != init_label_emb)}")

    for k, v in step_ret.items():
        print(f"{k}: {v}")

    print(f"Test Single Level Sampling with Label Conditioning")
    sample_labels = tf.random.uniform((4,), dtype=tf.int64, minval=0, maxval=9)
    print(f'[DEBUG] Labels: {sample_labels}')
    sampled_sequence, sampled_attn_w = prior.sample(4, y=sample_labels, return_attn_weights=True)

    print(sampled_sequence)

    for v, attn in sampled_attn_w.items():
        print(f"-------------{v}-------------")
        print(attn[0][0])  # (One Head, one sampled
