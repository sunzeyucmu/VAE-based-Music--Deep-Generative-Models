import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from utils import tf_utils


class VQVAEMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 test_dataset,
                 train_samples,
                 test_samples,
                 lr_scheduler,
                 ckpt_manager,
                 tb_summary_writer,
                 val_interval=2,
                 inspect_interval=5,
                 ckpt_interval=20,
                 sample_rate=3000,
                 **kwargs):
        super(VQVAEMonitor, self).__init__(**kwargs)
        self.val_interval = val_interval
        self.inspect_interval = inspect_interval
        self.val_dataset = test_dataset
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
        self.epoch = 0

    def on_batch_begin(self, batch, logs=None):
        tf.summary.experimental.set_step(self.step)

    def on_batch_end(self, batch, logs=None):

        self.step += 1


    def on_epoch_end(self, epoch, logs=None):
        # self.epoch += 1 # Start from 1
        if epoch != self.epoch:
            print(f"\n>>>>>>>>>>>>>>>>>>>>>Fit Epoch {epoch}, Monitor/Logger Epoch {self.epoch}\n")
        epoch = self.epoch

        # Checkpointing...
        if epoch % self.ckpt_interval == 0:
            ckpt_save_path = self.ckpt_manager.save()
            print(f'\nSaving checkpoint for epoch {epoch + 1} at {ckpt_save_path}\n')

        # Validation on test dataset
        if epoch % self.val_interval == 0:
            # Reset the metrics!!!
            print("\nResetting the metrics...")
            for m in self.model.metrics:
                m.reset_state()

            print("\n[DEBUG] This is Callback Monitor: End of Epoch", epoch)
            print("---------------------------Running Validation DataSet---------------------------")
            self.model.evaluate(self.val_dataset)
            with self.summary_writer.as_default():
                for m in self.model.metrics:
                    tf.summary.scalar(f'[val]{m.name}', m.result(), step=epoch)

        # Periodic Inspect
        if epoch % self.inspect_interval == 0:
            print(
                f"-------------------------------------Validate Test Samples Performance------------------------------------------")
            self.inspect_samples(self.test_samples, epoch, tag='val')


            # Training Samples
            print(
                f"-------------------------------------Validate Training Samples Performance--------------------------------------")
            self.inspect_samples(self.train_samples, epoch)
            # if isinstance(self.train_samples, tuple):
            #     print(f'---------Labels provided for Test Samples: {self.train_samples[1]}')
            #
            #
            # x_train_out = None
            # x_train_outs = []
            # for level in reversed(range(self.model.levels)):
            #     print(f"---------------Reconstruction at Level {level}/{self.model.levels}...-------------------------")
            #
            #     level_recons = tf_utils.generate_and_save_waves(self.model, 0, self.train_samples,
            #                                                     level=level,
            #                                                     if_decode=False,
            #                                                     if_sample=False)
            #     x_train_out = level_recons
            #
            #     x_train_outs.insert(0, level_recons)
            #
            # # TODO: 'x_in' as the target (encoded of raw input) of the prior model
            # self.log_input_output_audio(x_in=self.train_samples, x_out=x_train_out, x_outs=x_train_outs,
            #                             step=epoch)

        self.epoch += 1 # start from 0


    def inspect_samples(self, samples, step, tag='train'):
        if isinstance(samples, tuple):
            print(f'---------Labels provided for Test Samples: {samples[1]}')

        val_recons, metrics_dict = self.model(samples, training=False)

        print(
            ">>>>>>>>>>>Samples Loss: {}".format(
                "; ".join([f"{k}={list(map(lambda d: '{:.4f}'.format(d), v))}" for k, v in metrics_dict.items()])))
        # target = samples[0] if isinstance(samples, tuple) else samples
        # print(">>>>>>>>>>> Top 100 of Raw INPUT: ", target.squeeze()[:1, :100])
        # for val_recon in val_recons:
        #     print(">>>>>>>>>>> Top 100 of Recons: ", val_recon.numpy().squeeze()[:1, :100])

        x_val_out = None
        x_val_outs = []
        for level in reversed(range(self.model.levels)):
            print(f"---------------Reconstruction at Level {level}/{self.model.levels}...-------------------------")

            level_recons = tf_utils.generate_and_save_waves(self.model, 0, samples,
                                                            level=level,
                                                            if_decode=False,
                                                            if_sample=False)
            x_val_out = level_recons

            x_val_outs.insert(0, level_recons)

        # TODO: 'x_in' as the target (encoded of raw input) of the prior model
        self.log_input_output_audio(x_in=samples, x_out=x_val_out, x_outs=x_val_outs, tag=tag, step=step)


    def log_input_output_audio(self, x_in, x_out, x_outs, step, tag='train'):
        print(f"[DEBUG] Logging Audio files with {len(x_out)} level recons for Step: {self.step}.......")


        if isinstance(x_in, tuple):
            x_in = x_in[0]

        with self.summary_writer.as_default():
            tf.summary.audio(name=f"{tag}_audio_in", data=x_in, sample_rate=self.sr, step=step,
                             max_outputs=len(x_in))

            tf.summary.audio(name=f"{tag}_audio_out", data=x_out, sample_rate=self.sr, step=step,
                             max_outputs=len(x_out))

            for i in range(len(x_outs)):
                tf.summary.audio(name=f"{tag}_audio_level_{i}_out", data=x_outs[i], sample_rate=self.sr, step=step,
                                 max_outputs=len(x_outs[i]))
