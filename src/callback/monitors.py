import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from utils import tf_utils


class PriorMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 test_dataset,
                 train_samples,
                 test_samples,
                 lr_scheduler,
                 ckpt_manager,
                 tb_summary_writer,
                 val_interval=2,
                 decode_samples_interval=10,
                 sample_interval=50,
                 ckpt_interval=20,
                 sample_rate=3000,
                 **kwargs):
        super(PriorMonitor, self).__init__(**kwargs)
        self.val_interval = val_interval
        self.sample_interval = sample_interval
        self.val_dataset = test_dataset
        self.decode_samples_interval = decode_samples_interval
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.ckpt_manager = ckpt_manager
        self.summary_writer = tb_summary_writer
        self.ckpt_interval = ckpt_interval
        self.lr_scheduler = lr_scheduler

        self.sr = sample_rate

        # Capture Number of Training Steps/Iters so far
        # TODO: Restore this value to resume training?
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        tf.summary.experimental.set_step(self.step)

    def on_train_begin(self, logs=None):
        print("[DEBUG] start of training... Set TB Summary Writer Step as 0")
        # self.summary_writer.set_as_default()
        tf.summary.experimental.set_step(self.step)

    def on_batch_end(self, batch, logs=None):

        self.step += 1

        with self.summary_writer.as_default():
            tf.summary.scalar('[train_batch]loss', self.model.train_loss_tracker.result(), step=self.step)
            tf.summary.scalar('[train_batch]accuracy', self.model.train_accuracy_tracker.result(), step=self.step)
            tf.summary.scalar('[train_batch]perplexity/word', tf.math.exp(self.model.train_loss_tracker.result()),
                              step=self.step)
            tf.summary.scalar('[train_batch]lr', self.lr_scheduler(tf.cast(self.step, dtype=tf.float32)),
                              step=self.step)

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            tf.summary.scalar('[train]loss', self.model.train_loss_tracker.result(), step=epoch)
            tf.summary.scalar('[train]accuracy', self.model.train_accuracy_tracker.result(), step=epoch)
            tf.summary.scalar('[train]perplexity/word', tf.math.exp(self.model.train_loss_tracker.result()), step=epoch)

        if epoch % self.ckpt_interval == 0:
            ckpt_save_path = self.ckpt_manager.save()
            print(f'\nSaving checkpoint for epoch {epoch + 1} at {ckpt_save_path}\n')

        # Periodic Inspect
        if epoch % self.val_interval == 0:
            # Reset the metrics!!!
            print("\nResetting the metrics...")
            for m in self.model.metrics:
                m.reset_state()

            print("\n[DEBUG] This is Callback Monitor: End of Epoch", epoch)
            print("---------------------------Running Validation DataSet---------------------------")
            self.model.evaluate(self.val_dataset)
            with self.summary_writer.as_default():
                tf.summary.scalar('[test]loss', self.model.train_loss_tracker.result(), step=epoch)
                tf.summary.scalar('[test]accuracy', self.model.train_accuracy_tracker.result(), step=epoch)
                tf.summary.scalar('[test]perplexity/word', tf.math.exp(self.model.train_loss_tracker.result()),
                                  step=epoch)

        if epoch % self.decode_samples_interval == 0:
            print(
                f"-------------------------------------Validate Test Samples Performance------------------------------------------")
            if isinstance(self.test_samples, tuple):
                print(f'---------Labels provided for Test Samples: {self.test_samples[1]}')

            pred_logits_test, target_test, attn_weights_test, loss_test, accuracy_test = self.model(self.test_samples)

            print(
                f'Testing Samples Loss {loss_test:.4f}; Perplexity (exp of loss_per_word): {tf.math.exp(loss_test):.5f}; Accuracy {accuracy_test:.4f}')
            print(">>>>>>>>>>> Top 100 of Test Target: ", target_test[0][:100])
            print(">>>>>>>>>>> Top 100 of Test Preds: ", tf.argmax(pred_logits_test, axis=2)[0][:100])

            for k, v in attn_weights_test.items():
                print(k)
                print(v.shape)
                tf_utils.plot_attention_weights(v[0])

            test_recons = tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.test_samples,
                                                           level=self.model.level,
                                                           if_decode=True,
                                                           latent_code=tf.argmax(pred_logits_test, axis=-1),
                                                           if_sample=False)

            # TODO: 'x_in' as the target (encoded of raw input) of the prior model
            self.log_input_output_audio(x_in=self.test_samples, x_out=test_recons, if_sample=False, tag='val', step=epoch)

            # Training Samples
            print(
                f"-------------------------------------Validate Training Samples Performance--------------------------------------")
            if isinstance(self.train_samples, tuple):
                print(f'---------Labels provided for Test Samples: {self.train_samples[1]}')

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
                sampled_recons, sampled_codes = tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level,
                                                 if_decode=True,
                                                 latent_code=tf.argmax(pred_logits, axis=-1), if_sample=True,
                                                 prior_model=self.model)
                ## TODO: Multiple Sample Modes
                self.log_input_output_audio(x_in=None, x_out=sampled_recons, if_sample=True, step=epoch)

            else:
                train_recons = tf_utils.generate_and_save_waves(self.model.vqvae, 0, self.train_samples, level=self.model.level,
                                                 if_decode=True,
                                                 latent_code=tf.argmax(pred_logits, axis=-1), if_sample=False)

                self.log_input_output_audio(x_in=self.train_samples, x_out=train_recons, if_sample=False, step=epoch)

            return pred_logits, attn_weights

    def log_input_output_audio(self, x_in, x_out, step, tag='train', if_sample=False, sample_mode=0):
        print(f"[DEBUG] Logging Audio files (IF SAMPLE: {if_sample}) for Step: {self.step}.......")

        if if_sample:
            with self.summary_writer.as_default():
                tf.summary.audio(name=f"{tag}_sample_mode{sample_mode}", data=x_out, sample_rate=self.sr, step=step,
                                 max_outputs=len(x_out))

        else:
            if isinstance(x_in, tuple):
                x_in = x_in[0]

            with self.summary_writer.as_default():
                tf.summary.audio(name=f"{tag}_audio_in", data=x_in, sample_rate=self.sr, step=step,
                                 max_outputs=len(x_in))

                tf.summary.audio(name=f"{tag}_audio_out", data=x_out, sample_rate=self.sr, step=step,
                                 max_outputs=len(x_out))
