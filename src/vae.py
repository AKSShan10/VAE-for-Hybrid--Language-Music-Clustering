# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras import layers


# # class Sampling(layers.Layer):
# #     """Reparameterization trick: z = mu + eps * sigma."""
# #     def call(self, inputs):
# #         mu, logvar = inputs
# #         eps = tf.random.normal(shape=tf.shape(mu))
# #         return mu + tf.exp(0.5 * logvar) * eps


# # def build_vae(input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
# #     # Encoder
# #     x_in = keras.Input(shape=(input_dim,), name="x_in")
# #     h = layers.Dense(hidden_dim, activation="relu")(x_in)
# #     h = layers.Dense(hidden_dim, activation="relu")(h)
# #     mu = layers.Dense(latent_dim, name="mu")(h)
# #     logvar = layers.Dense(latent_dim, name="logvar")(h)
# #     z = Sampling()([mu, logvar])
# #     encoder = keras.Model(x_in, [mu, logvar, z], name="encoder")

# #     # Decoder
# #     z_in = keras.Input(shape=(latent_dim,), name="z_in")
# #     h2 = layers.Dense(hidden_dim, activation="relu")(z_in)
# #     h2 = layers.Dense(hidden_dim, activation="relu")(h2)
# #     x_out = layers.Dense(input_dim, name="x_out")(h2)
# #     decoder = keras.Model(z_in, x_out, name="decoder")

# #     return encoder, decoder


# # class VAE(keras.Model):
# #     def __init__(self, encoder, decoder, beta: float = 0.3, **kwargs):
# #         super().__init__(**kwargs)
# #         self.encoder = encoder
# #         self.decoder = decoder
# #         self.beta = beta
# #         self.total_loss_tracker = keras.metrics.Mean(name="loss")
# #         self.recon_loss_tracker = keras.metrics.Mean(name="recon")
# #         self.kl_loss_tracker = keras.metrics.Mean(name="kl")

# #     @property
# #     def metrics(self):
# #         return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

# #     def train_step(self, data):
# #         x = data[0] if isinstance(data, (tuple, list)) else data
# #         with tf.GradientTape() as tape:
# #             mu, logvar, z = self.encoder(x, training=True)
# #             x_hat = self.decoder(z, training=True)

# #             recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))  # SSE per sample
# #             kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))
# #             total = recon + self.beta * kl

# #         grads = tape.gradient(total, self.trainable_weights)
# #         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

# #         self.total_loss_tracker.update_state(total)
# #         self.recon_loss_tracker.update_state(recon)
# #         self.kl_loss_tracker.update_state(kl)

# #         return {"loss": self.total_loss_tracker.result(),
# #                 "recon": self.recon_loss_tracker.result(),
# #                 "kl": self.kl_loss_tracker.result()}



# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# class Sampling(layers.Layer):
#     """Reparameterization trick: z = mu + eps * sigma."""
#     def call(self, inputs):
#         mu, logvar = inputs
#         eps = tf.random.normal(shape=tf.shape(mu))
#         return mu + tf.exp(0.5 * logvar) * eps


# def build_vae(input_dim: int, latent_dim: int = 32, hidden_dim: int = 256, dropout: float = 0.1):
#     # Encoder
#     x_in = keras.Input(shape=(input_dim,), name="x_in")

#     h = layers.Dense(hidden_dim)(x_in)
#     h = layers.BatchNormalization()(h)
#     h = layers.Activation("relu")(h)
#     h = layers.Dropout(dropout)(h)

#     h = layers.Dense(hidden_dim)(h)
#     h = layers.BatchNormalization()(h)
#     h = layers.Activation("relu")(h)
#     h = layers.Dropout(dropout)(h)

#     mu = layers.Dense(latent_dim, name="mu")(h)
#     logvar = layers.Dense(latent_dim, name="logvar")(h)
#     z = Sampling()([mu, logvar])
#     encoder = keras.Model(x_in, [mu, logvar, z], name="encoder")

#     # Decoder
#     z_in = keras.Input(shape=(latent_dim,), name="z_in")

#     h2 = layers.Dense(hidden_dim)(z_in)
#     h2 = layers.BatchNormalization()(h2)
#     h2 = layers.Activation("relu")(h2)
#     h2 = layers.Dropout(dropout)(h2)

#     h2 = layers.Dense(hidden_dim)(h2)
#     h2 = layers.BatchNormalization()(h2)
#     h2 = layers.Activation("relu")(h2)
#     h2 = layers.Dropout(dropout)(h2)

#     x_out = layers.Dense(input_dim, name="x_out")(h2)
#     decoder = keras.Model(z_in, x_out, name="decoder")

#     return encoder, decoder


# class VAE(keras.Model):
#     def __init__(self, encoder, decoder, beta: float = 0.3, kl_warmup_steps: int = 2000, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder

#         # Target KL weight (beta-VAE style, but still "basic VAE" friendly)
#         self.beta = float(beta)

#         # KL warm-up: linearly ramps KL weight from 0 to beta over these steps
#         self.kl_warmup_steps = int(kl_warmup_steps)
#         self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

#         self.total_loss_tracker = keras.metrics.Mean(name="loss")
#         self.recon_loss_tracker = keras.metrics.Mean(name="recon")
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl")
#         self.kl_weight_tracker = keras.metrics.Mean(name="kl_weight")

#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#             self.recon_loss_tracker,
#             self.kl_loss_tracker,
#             self.kl_weight_tracker,
#         ]

#     def _current_kl_weight(self):
#         # Linear warm-up: min(beta, beta * step / warmup_steps)
#         step = tf.cast(self.step_counter, tf.float32)
#         warm = tf.cast(self.kl_warmup_steps, tf.float32)
#         w = self.beta * tf.minimum(1.0, step / tf.maximum(1.0, warm))
#         return w

#     def train_step(self, data):
#         x = data[0] if isinstance(data, (tuple, list)) else data
#         self.step_counter.assign_add(1)

#         with tf.GradientTape() as tape:
#             mu, logvar, z = self.encoder(x, training=True)
#             x_hat = self.decoder(z, training=True)

#             # Use per-dimension MSE (stable across feature sizes)
#             recon = tf.reduce_mean(tf.reduce_mean(tf.square(x - x_hat), axis=1))

#             kl = -0.5 * tf.reduce_mean(
#                 tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
#             )

#             kl_w = self._current_kl_weight()
#             total = recon + kl_w * kl

#         grads = tape.gradient(total, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

#         self.total_loss_tracker.update_state(total)
#         self.recon_loss_tracker.update_state(recon)
#         self.kl_loss_tracker.update_state(kl)
#         self.kl_weight_tracker.update_state(kl_w)

#         return {
#             "loss": self.total_loss_tracker.result(),
#             "recon": self.recon_loss_tracker.result(),
#             "kl": self.kl_loss_tracker.result(),
#             "kl_weight": self.kl_weight_tracker.result(),
#         }



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + eps * sigma."""
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps


def build_vae(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    dropout: float = 0.05,
    l2_weight: float = 1e-4,
):
    reg = keras.regularizers.l2(l2_weight) if l2_weight and l2_weight > 0 else None

    # Encoder
    x_in = keras.Input(shape=(input_dim,), name="x_in")

    h = layers.Dense(hidden_dim, kernel_regularizer=reg)(x_in)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(dropout)(h)

    h = layers.Dense(hidden_dim, kernel_regularizer=reg)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(dropout)(h)

    mu = layers.Dense(latent_dim, name="mu")(h)
    logvar = layers.Dense(latent_dim, name="logvar")(h)
    z = Sampling()([mu, logvar])
    encoder = keras.Model(x_in, [mu, logvar, z], name="encoder")

    # Decoder
    z_in = keras.Input(shape=(latent_dim,), name="z_in")

    h2 = layers.Dense(hidden_dim, kernel_regularizer=reg)(z_in)
    h2 = layers.BatchNormalization()(h2)
    h2 = layers.Activation("relu")(h2)
    h2 = layers.Dropout(dropout)(h2)

    h2 = layers.Dense(hidden_dim, kernel_regularizer=reg)(h2)
    h2 = layers.BatchNormalization()(h2)
    h2 = layers.Activation("relu")(h2)
    h2 = layers.Dropout(dropout)(h2)

    x_out = layers.Dense(input_dim, name="x_out")(h2)
    decoder = keras.Model(z_in, x_out, name="decoder")

    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta: float = 0.5, kl_warmup_steps: int = 600, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.beta = float(beta)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl")
        self.kl_weight_tracker = keras.metrics.Mean(name="kl_weight")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.kl_weight_tracker,
        ]

    def _current_kl_weight(self):
        step = tf.cast(self.step_counter, tf.float32)
        warm = tf.cast(self.kl_warmup_steps, tf.float32)
        return self.beta * tf.minimum(1.0, step / tf.maximum(1.0, warm))

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        self.step_counter.assign_add(1)

        with tf.GradientTape() as tape:
            mu, logvar, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)

            # per-dimension MSE
            recon = tf.reduce_mean(tf.reduce_mean(tf.square(x - x_hat), axis=1))

            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )

            kl_w = self._current_kl_weight()
            total = recon + kl_w * kl

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        self.kl_weight_tracker.update_state(kl_w)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon": self.recon_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight_tracker.result(),
        }
