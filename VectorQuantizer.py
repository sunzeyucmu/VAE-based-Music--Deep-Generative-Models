import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, codebook_usage_threshold=1.0, decay_rate=0.99, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # beta is the hyper-param to control the reluctance of changing the codebook latent code accroding to encoder output (scaled commitment loss)
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )
        self.codebook_usage_threshold = codebook_usage_threshold

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        ## Latent Embedding Space: K x D/L
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        ) # The Model Weights


        # EMA for codebook update; replacement for codebook loss

        ## Record Embedding Usage
        self.N_t = tf.zeros([num_embeddings,], dtype=tf.float32)

        ## Metrics
        self.usage_tracker = keras.metrics.Mean(name="codebook_usage")

    @property
    def metrics(self):
        return [
            self.usage_tracker,
        ]

    def call(self, x, debug=False):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        # input x: 2D: (B, H, W, L); 1D: (B, T, L) -flatten-> (N, L)
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        ## (N, )
        encoding_indices = self.get_code_indices(flattened)
        # (N, K): get one-hot embedding for each instance/row of N
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # (N, L): get the correcponding embedding for each instance of N from the codebook/E
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        # recover the previous flattened input to (B, H, W, L) in Image/Conv2D case
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        '''
        Straight-through estimator： 
            Training/BackProp: During backpropagation, (quantized - x) won't be included in the computation graph 
            and gradients obtaind for quantized will be copied&&pasted for inputs
        '''
        quantized = x + tf.stop_gradient(quantized - x)

        ## EMA (Exponential Moving Average) calculation
        self.N_t = tf.reduce_sum(encodings, axis=0) # (K, )


        ## METRCIS Tracking
        codebook_usage = tf.reduce_sum(tf.cast(self.N_t >= self.codebook_usage_threshold, dtype=tf.float32))
        self.usage_tracker.update_state(codebook_usage)

        ## DEBUG
        if debug:
          print("VQ input (Encoder Output): ", x)
          print("VQ output: ", quantized)
        return quantized , encoding_indices

    '''
    Output shape: (N, )
    '''
    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        # (N, K)
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        # (N, K)
        distances = (
            # for each row/instance i of the flattened input
            # after broadcasting
            # zei^2 + ek^2 - 2(zei^Tek)
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0) # broadcast on (N, 1) and (1, K)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_usage_count(self):
        return self.N_t


if __name__ == '__main__':
    print('VQ module')

    latent_dim = 2
    # the latent/embedding width needs to be shared between a.encoder output b. vq output and c. decoder input;
    # should be treated as $$model width
    VQ = VectorQuantizer(num_embeddings=6, embedding_dim=latent_dim)
    print(VQ.get_usage_count())
    test_VQ_out, test_latent_idx = VQ(tf.random.normal([32, 100, latent_dim]))
    print(VQ.metrics[0].result())

    # Trainable Variables
    print(VQ.trainable_variables) # Embeddings