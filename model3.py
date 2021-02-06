import tensorflow as tf
from utils import *

# noinspection PyAttributeOutsideInit
class GaussianAttention(tf.keras.layers.Layer):
    def __init__(self, kmixtures, ascii_steps, char_vec_len, window_w_initializer, window_b_initializer):
        super(GaussianAttention, self).__init__()
        self.kmixtures = kmixtures
        self.ascii_steps = ascii_steps
        self.char_vec_len = char_vec_len
        self.window_w_initializer = window_w_initializer
        self.window_b_initializer = window_b_initializer

    def build(self, input_shape):
        batch = input_shape[0]
        hidden = input_shape[2]
        n_out = 3 * self.kmixtures
        self.init_kappa = self.add_weight("init_kappa", shape=[batch, self.kmixtures, 1])
        self.char_seq = self.add_weight("char_seq", shape=[batch, self.ascii_steps, self.char_vec_len])
        self.window_w = self.add_weight("window_w", shape=[hidden, n_out], initializer=self.window_w_initializer)
        self.window_b = self.add_weight("window_b", shape=[n_out], initializer=self.window_b_initializer)

    def call(self, input0, **kwargs):
        assoc = tf.unstack(kwargs['original'], axis=1)
        result = tf.unstack(input0, axis=1)
        prev_kappa = self.init_kappa
        for i in range(len(result)):
            [alpha, beta, new_kappa] = self.get_window_params(result[i], prev_kappa)
            window, phi = self.get_window(alpha, beta, new_kappa, self.char_seq)
            result[i] = tf.concat((result[i], window, assoc[i]), 1)
            prev_kappa = new_kappa
        return tf.stack(result, axis=1)

    # ----- build the gaussian character window
    def get_window(self, alpha, beta, kappa, c):
        # phi -> [? x 1 x ascii_steps] and is a tf matrix
        # c -> [? x ascii_steps x alphabet] and is a tf matrix
        ascii_steps = c.get_shape()[1]  # number of items in sequence
        phi = self.get_phi(ascii_steps, alpha, beta, kappa)
        window = tf.matmul(phi, c)
        window = tf.squeeze(window, [1])  # window ~ [?,alphabet]
        return window, phi

    # get phi for all t,u (returns a [1 x tsteps] matrix) that defines the window
    def get_phi(self, ascii_steps, alpha, beta, kappa):
        # alpha, beta, kappa -> [?,kmixtures,1] and each is a tf variable
        u = np.linspace(0, ascii_steps - 1, ascii_steps)  # weight all the U items in the sequence
        kappa_term = tf.square(tf.subtract(kappa, u))
        exp_term = tf.multiply(-beta, kappa_term)
        phi_k = tf.multiply(alpha, tf.exp(exp_term))
        return tf.reduce_sum(input_tensor=phi_k, axis=1, keepdims=True)  # phi ~ [?,1,ascii_steps]

    def get_window_params(self, out_cell0, prev_kappa):
        abk_hats = tf.add(tf.matmul(out_cell0, self.window_w), self.window_b)  # abk_hats ~ [?,n_out]
        # abk_hats ~ [?,n_out] = "alpha, beta, kappa hats"
        abk = tf.exp(tf.reshape(abk_hats, [-1, 3 * self.kmixtures, 1]))
        alpha, beta, kappa = tf.split(abk, 3, 1)  # alpha_hat, etc ~ [?,kmixtures]
        kappa = kappa + prev_kappa
        return alpha, beta, kappa  # each ~ [?,kmixtures,1]


# noinspection PyAttributeOutsideInit
class MDN(tf.keras.layers.Layer):
    def __init__(self, rnn_size, nmixtures, initializer):
        super(MDN, self).__init__()
        self.rnn_size = rnn_size
        self.nmixtures = nmixtures
        self.initializer = initializer

    def build(self, input_shape):
        n_out = 1 + self.nmixtures * 6  # params = end_of_stroke + 6 parameters per Gaussian
        self.mdn_w = self.add_weight("output_w", shape=[self.rnn_size, n_out], initializer=self.initializer)
        self.mdn_b = self.add_weight("output_b", shape=[n_out], initializer=self.initializer)

    def call(self, input0, **kwargs):
        flattened = tf.reshape(tf.concat(input0, 0), [-1, self.rnn_size])  # concat outputs for efficiency
        dense = tf.add(tf.matmul(flattened, self.mdn_w), self.mdn_b)
        # now transform dense NN outputs into params for MDN
        [self.eos, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho] = self.get_mdn_coef(dense)
        return input0

    def get_mdn_coef(self, Z):
        # returns the tf slices containing mdn dist params (eq 18...23 of http://arxiv.org/abs/1308.0850)
        eos_hat = Z[:, 0:1]  # end of sentence tokens
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(Z[:, 1:], 6, 1)
        # these are useful for bias method during sampling
        self.pi_hat, self.sigma1_hat, self.sigma2_hat = pi_hat, sigma1_hat, sigma2_hat
        eos = tf.sigmoid(-1 * eos_hat)  # technically we gained a negative sign
        pi = tf.nn.softmax(pi_hat)  # softmax z_pi:
        mu1 = mu1_hat
        mu2 = mu2_hat  # leave mu1, mu2 as they are
        sigma1 = tf.exp(sigma1_hat)
        sigma2 = tf.exp(sigma2_hat)  # exp for sigmas
        rho = tf.tanh(rho_hat)  # tanh for rho (squish between -1 and 1)
        return [eos, pi, mu1, mu2, sigma1, sigma2, rho]


class Model:
    def __init__(self, args, logger):
        self.logger = logger

        # ----- transfer some of the args params over to the model

        # model params
        self.rnn_size = args.rnn_size
        self.train = args.train
        self.nmixtures = args.nmixtures
        self.kmixtures = args.kmixtures
        self.batch_size = args.batch_size if self.train else 1  # training/sampling specific
        self.tsteps = args.tsteps if self.train else 1  # training/sampling specific
        self.alphabet = args.alphabet
        # training params
        self.dropout = args.dropout
        self.grad_clip = args.grad_clip
        # misc
        self.tsteps_per_ascii = args.tsteps_per_ascii
        self.data_dir = args.data_dir

        self.logger.write('\tusing alphabet{}'.format(self.alphabet))
        self.char_vec_len = len(self.alphabet) + 1  # plus one for <UNK> token
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)

        self.graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)
        self.window_b_initializer = tf.keras.initializers.TruncatedNormal(mean=-3.0, stddev=0.25)

        # build the network
        network_input = tf.keras.layers.Input(shape=(self.tsteps, 3,), batch_size=self.batch_size)

        cell1 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=self.graves_initializer, dropout=self.dropout
        )
        cell2 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=self.graves_initializer, dropout=self.dropout
        )
        cell3 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=self.graves_initializer, dropout=self.dropout
        )

        out_lstm1 = cell1(network_input)
        out_attention = GaussianAttention(
            self.kmixtures, self.ascii_steps, self.char_vec_len,
            self.graves_initializer, self.window_b_initializer
        )(out_lstm1, original=network_input)
        out_lstm2 = cell2(out_attention)
        out_lstm3 = cell3(out_lstm2)
        out_mdn = MDN(args.rnn_size, self.nmixtures, self.graves_initializer)(out_lstm3)
        model = tf.keras.Model([network_input], out_mdn)

        model.summary()

        # reshape target data (as we did the input data)
        flat_target_data = tf.reshape(self.target_data, [-1, 3])
        [x1_data, x2_data, eos_data] = tf.split(flat_target_data, 3, 1)  # we might as well split these now

        loss = self.get_loss(self.pi, x1_data, x2_data, eos_data, self.mu1, self.mu2, self.sigma1, self.sigma2,
                             self.rho,
                             self.eos)
        self.cost = loss / (self.batch_size * self.tsteps)

        # ----- bring together all variables and prepare for training
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.decay = tf.Variable(0.0, trainable=False)
        self.momentum = tf.Variable(0.0, trainable=False)

        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ys=self.cost, xs=tvars), self.grad_clip)

        if args.optimizer == 'adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif args.optimizer == 'rmsprop':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay,
                                                                 momentum=self.momentum)
        else:
            raise ValueError("Optimizer type not recognized")
        self.train_op = self.optimizer.apply_gradients(list(zip(grads, tvars)))

        # ----- some TensorFlow I/O
        self.sess = tf.compat.v1.InteractiveSession()
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # ----- for restoring previous models
    def try_load_model(self, save_path):
        load_was_success = True  # yes, I'm being optimistic
        global_step = 0
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            self.logger.write("no saved model to load. starting new session")
            load_was_success = False
        else:
            self.logger.write("loaded model: {}".format(load_path))
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            global_step = int(load_path.split('-')[-1])
        return load_was_success, global_step

    # Util methods for building the model

    # ----- build mixture density cap on top of second recurrent cell
    def gaussian2d(self, x1, x2, mu1, mu2, s1, s2, rho):
        # define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
        x_mu1 = tf.subtract(x1, mu1)
        x_mu2 = tf.subtract(x2, mu2)
        Z = tf.square(tf.divide(x_mu1, s1)) + \
            tf.square(tf.divide(x_mu2, s2)) - \
            2 * tf.divide(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
        rho_square_term = 1 - tf.square(rho)
        power_e = tf.exp(tf.divide(-Z, 2 * rho_square_term))
        regularize_term = 2 * np.pi * tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
        gaussian = tf.divide(power_e, regularize_term)
        return gaussian

    def get_loss(self, pi, x1_data, x2_data, eos_data, mu1, mu2, sigma1, sigma2, rho, eos):
        # define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
        gaussian = self.gaussian2d(x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho)
        term1 = tf.multiply(gaussian, pi)
        term1 = tf.reduce_sum(input_tensor=term1, axis=1, keepdims=True)  # do inner summation
        term1 = -tf.math.log(tf.maximum(term1, 1e-20))  # some errors are zero -> numerical errors.

        term2 = tf.multiply(eos, eos_data) + tf.multiply(1 - eos,
                                                         1 - eos_data)  # modified Bernoulli -> eos probability
        term2 = -tf.math.log(term2)  # negative log error gives loss

        return tf.reduce_sum(input_tensor=term1 + term2)  # do outer summation
