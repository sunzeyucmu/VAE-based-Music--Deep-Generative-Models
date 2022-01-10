import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class VectorQuantizer(layers.Layer):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 beta=0.25,
                 codebook_usage_threshold=1.0,
                 decay_rate=0.99,
                 level=0, **kwargs):
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
        # self.embeddings = tf.Variable(
        #     initial_value=w_init(
        #         shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
        #     ),
        #     trainable=True,
        #     name="embeddings_vqvae",
        # ) # The Model Weights


        # EMA for codebook update; replacement for codebook loss
        self.gamma = decay_rate
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=False,
            name="embeddings_vqvae",
        ) # Non-trainable Model Weights@

        # (L, K): running sum of encoder_outputs (ez) clustered around each codebook vector
        # self.m_t = self.embeddings
        self.m_t = tf.Variable(
            initial_value=self.embeddings,
            trainable=False,
        )
        ## Record Embedding Usage
        # self.N_t = tf.zeros([num_embeddings,], dtype=tf.float32) # (K, ): usage count of codebook vectors
        # (K, ): usage count of codebook vectors
        # $ equal start for all codebook vectors
        # self.N_t = tf.ones([num_embeddings, ], dtype=tf.float32)
        self.N_t = tf.Variable(
            initial_value=tf.ones([num_embeddings, ], dtype=tf.float32),
            trainable=False,
        )
        ## Metrics
        self.batch_usage_tracker = keras.metrics.Mean(name="[{}]batch_codebook_usage".format(level))
        self.usage_tracker = keras.metrics.Mean(name="[{}]codebook_usage".format(level))
        self.entropy_tracker = keras.metrics.Mean(name="[{}]codebook_entropy".format(level))

    @property
    def metrics(self):
        return [
            self.batch_usage_tracker,
            self.usage_tracker,
            self.entropy_tracker
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
        # EMA, replacement for codebook loss
        # codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        # self.add_loss(commitment_loss + codebook_loss)
        '''
        This property is reset at the start of every __call__() to the top-level layer, 
        so that layer.losses always contains the loss values created during the last forward pass.
        '''
        self.add_loss(commitment_loss)

        '''
        Straight-through estimator： 
            Training/BackProp: During backpropagation, (quantized - x) won't be included in the computation graph 
            and gradients obtaind for quantized will be copied&&pasted for inputs
        '''
        quantized = x + tf.stop_gradient(quantized - x)

        ## EMA (Exponential Moving Average) calculation
        '''
        [L, NT] x [NT, K] -> [L, K]: for each col (1/k codebook embedding ek), sum up all encoder outputs (zk,1, ... zk,nk)
        which are closest to ek; for new centre computation. (e.g. K-Means....)
        '''
        m_t_ = tf.matmul(flattened, encodings, transpose_a=True) # m_(t)
        N_t_ = tf.reduce_sum(encodings, axis=0) # (K, ): N_(t)

        # moving average of mi(t)
        # self.m_t = self.gamma * self.m_t + (1. - self.gamma) * m_t_
        self.m_t.assign(self.gamma * self.m_t + (1. - self.gamma) * m_t_)
        # moving average of Ni(t)
        # self.N_t = self.gamma * self.N_t + (1. - self.gamma) * N_t_  # k_bins
        self.N_t.assign(self.gamma * self.N_t + (1. - self.gamma) * N_t_)

        usage = tf.reshape(tf.cast(self.N_t >= self.codebook_usage_threshold, dtype=tf.float32), [1, self.num_embeddings])
        # TODO: assume NT > K here...
        # random_codes = tf.transpose(tf.random.shuffle(flattened)[:self.num_embeddings]) # (L, K)
        ## Tile X first incase NT < K
        random_codes = tf.transpose(tf.random.shuffle(self._tile(flattened))[:self.num_embeddings])  # (L, K)
        reset_codes = (1.0 - usage) * random_codes
        # TODO: re-randomize below threshold vectors to current encoder output

        # !NAN prevention: running count could go zero... clip it
        # self.N_t.assign(tf.clip_by_value(self.N_t, 1e-8, 1e+8))
        # self.embeddings.assign(self.m_t / tf.reshape(tf.clip_by_value(self.N_t, 1e-8, 1e+8), [1, self.num_embeddings]))
        self.embeddings.assign(usage * (self.m_t / tf.reshape(tf.clip_by_value(self.N_t, 1e-8, 1e+8), [1, self.num_embeddings]))
                               + reset_codes)

        # self.N_t = tf.reduce_sum(encodings, axis=0) # (K, )

        ## METRCIS Tracking
        # usage for current batch
        cur_codebook_usage = tf.reduce_sum(tf.cast(N_t_ >= self.codebook_usage_threshold, dtype=tf.float32))
        # running average usage monitoring (Codebook Collapse...)
        codebook_usage = tf.reduce_sum(tf.cast(self.N_t >= self.codebook_usage_threshold, dtype=tf.float32))
        self.batch_usage_tracker.update_state(cur_codebook_usage)
        self.usage_tracker.update_state(codebook_usage)
        # Entropy (Codebook vector diversity)
        code_prob = N_t_ / tf.reduce_sum(N_t_)
        code_entropy = -tf.reduce_sum(code_prob * tf.math.log(code_prob + 1e-8))
        self.entropy_tracker.update_state(code_entropy)

        ## DEBUG
        if debug:
          print("VQ input (Encoder Output): ", x)
          print("VQ output: ", quantized)
        return quantized, encoding_indices

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

    def _tile(self, x):
        # SafeGuard when NT < K
        nt = tf.shape(x)[0]
        ret = x
        if nt < self.num_embeddings:
            repeats = (self.num_embeddings + nt - 1) // nt
            ret = tf.tile(x, [repeats, 1])

        return ret




if __name__ == '__main__':
    print('VQ module')

    latent_dim = 2
    # the latent/embedding width needs to be shared between a.encoder output b. vq output and c. decoder input;
    # should be treated as $$model width
    VQ = VectorQuantizer(num_embeddings=6, embedding_dim=latent_dim)
    print(VQ.get_usage_count())
    print(VQ.m_t)
    print("Initial CodeBook: ", VQ.embeddings)
    test_VQ_out, test_latent_idx = VQ(tf.random.normal([32, 100, latent_dim]))
    print("N_t: ", VQ.get_usage_count())
    print("m_t: ", VQ.m_t)
    print("Updated embeddings (e_t): ", VQ.embeddings)
    print("Metrics: ", {m.name:m.result() for m in VQ.metrics})

    # Trainable Variables
    print("Trainable Variables: ", VQ.trainable_variables) # Embeddings

    # Test NT < K
    # VQ1 = VectorQuantizer(num_embeddings=5000, embedding_dim=latent_dim)