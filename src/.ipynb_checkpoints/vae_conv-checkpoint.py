# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# class Sampling(layers.Layer):
#     def call(self, inputs):
#         mu, logvar = inputs
#         eps = tf.random.normal(shape=tf.shape(mu))
#         return mu + tf.exp(0.5 * logvar) * eps


# def build_conv_vae(
#     input_shape=(64, 256, 1),
#     latent_dim=32,
#     beta=0.5,
#     kl_warmup_steps=800,   # reasonable for ~1000 samples, bs~32
# ):
#     # Encoder
#     x_in = keras.Input(shape=input_shape, name="spec_in")

#     x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x_in)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)

#     shape_before_flatten = x.shape[1:]  # (h,w,c)
#     x = layers.Flatten()(x)
#     x = layers.Dense(256, activation="relu")(x)

#     mu = layers.Dense(latent_dim, name="mu")(x)
#     logvar = layers.Dense(latent_dim, name="logvar")(x)
#     z = Sampling()([mu, logvar])

#     encoder = keras.Model(x_in, [mu, logvar, z], name="conv_encoder")

#     # Decoder
#     z_in = keras.Input(shape=(latent_dim,), name="z_in")
#     x = layers.Dense(256, activation="relu")(z_in)

#     flat_dim = int(shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2])
#     x = layers.Dense(flat_dim, activation="relu")(x)
#     x = layers.Reshape(shape_before_flatten)(x)

#     x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)

#     x_out = layers.Conv2D(1, 3, padding="same", activation=None, name="spec_out")(x)
#     decoder = keras.Model(z_in, x_out, name="conv_decoder")

#     class ConvVAE(keras.Model):
#         def __init__(self, encoder, decoder, beta, kl_warmup_steps, **kwargs):
#             super().__init__(**kwargs)
#             self.encoder = encoder
#             self.decoder = decoder
#             self.beta = float(beta)
#             self.kl_warmup_steps = int(kl_warmup_steps)
#             self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

#             self.total_loss_tracker = keras.metrics.Mean(name="loss")
#             self.recon_loss_tracker = keras.metrics.Mean(name="recon")
#             self.kl_loss_tracker = keras.metrics.Mean(name="kl")
#             self.kl_weight_tracker = keras.metrics.Mean(name="kl_weight")

#         @property
#         def metrics(self):
#             return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, self.kl_weight_tracker]

#         def _kl_weight(self):
#             step = tf.cast(self.step_counter, tf.float32)
#             warm = tf.cast(self.kl_warmup_steps, tf.float32)
#             return self.beta * tf.minimum(1.0, step / tf.maximum(1.0, warm))

#         def train_step(self, data):
#             x = data[0] if isinstance(data, (tuple, list)) else data
#             self.step_counter.assign_add(1)

#             with tf.GradientTape() as tape:
#                 mu, logvar, z = self.encoder(x, training=True)
#                 x_hat = self.decoder(z, training=True)

#                 recon = tf.reduce_mean(tf.reduce_mean(tf.square(x - x_hat), axis=[1, 2, 3]))
#                 kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))

#                 kl_w = self._kl_weight()
#                 total = recon + kl_w * kl

#             grads = tape.gradient(total, self.trainable_weights)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

#             self.total_loss_tracker.update_state(total)
#             self.recon_loss_tracker.update_state(recon)
#             self.kl_loss_tracker.update_state(kl)
#             self.kl_weight_tracker.update_state(kl_w)

#             return {
#                 "loss": self.total_loss_tracker.result(),
#                 "recon": self.recon_loss_tracker.result(),
#                 "kl": self.kl_loss_tracker.result(),
#                 "kl_weight": self.kl_weight_tracker.result(),
#             }

#     model = ConvVAE(encoder, decoder, beta=beta, kl_warmup_steps=kl_warmup_steps)
#     return encoder, decoder, model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        # logvar clipping can also be done here, but we do it in train_step for clarity
        return mu + tf.exp(0.5 * logvar) * eps


def build_conv_vae(
    input_shape=(64, 256, 1),
    latent_dim=32,
    beta=0.5,
    kl_warmup_steps=800,
):
    # ---------------- Encoder ----------------
    x_in = keras.Input(shape=input_shape, name="spec_in")

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    shape_before_flatten = x.shape[1:]  # (h, w, c)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    mu = layers.Dense(latent_dim, name="mu")(x)
    logvar = layers.Dense(latent_dim, name="logvar")(x)
    z = Sampling()([mu, logvar])

    encoder = keras.Model(x_in, [mu, logvar, z], name="conv_encoder")

    # ---------------- Decoder ----------------
    z_in = keras.Input(shape=(latent_dim,), name="z_in")
    x = layers.Dense(256, activation="relu")(z_in)

    flat_dim = int(shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2])
    x = layers.Dense(flat_dim, activation="relu")(x)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Linear output is OK because input spectrogram is normalized.
    # If you still see instability, you can constrain it with tanh:
    # x_out = layers.Conv2D(1, 3, padding="same", activation="tanh", name="spec_out")(x)
    x_out = layers.Conv2D(1, 3, padding="same", activation=None, name="spec_out")(x)

    decoder = keras.Model(z_in, x_out, name="conv_decoder")

    # ---------------- VAE wrapper ----------------
    class ConvVAE(keras.Model):
        def __init__(self, encoder, decoder, beta, kl_warmup_steps, **kwargs):
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
                self.kl_weight_tracker
            ]

        def _kl_weight(self):
            step = tf.cast(self.step_counter, tf.float32)
            warm = tf.cast(self.kl_warmup_steps, tf.float32)
            return self.beta * tf.minimum(1.0, step / tf.maximum(1.0, warm))

        def train_step(self, data):
            x = data[0] if isinstance(data, (tuple, list)) else data
            self.step_counter.assign_add(1)

            # Safety: ensure inputs are finite
            x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

            with tf.GradientTape() as tape:
                mu, logvar, z = self.encoder(x, training=True)

                # CRITICAL: clip logvar to prevent exp(logvar) overflow
                logvar = tf.clip_by_value(logvar, -10.0, 10.0)

                x_hat = self.decoder(z, training=True)

                # Optional: clip x_hat as an extra stability guard (enable if needed)
                # x_hat = tf.clip_by_value(x_hat, -5.0, 5.0)

                # Recon loss: mean squared error per sample (stable)
                recon = tf.reduce_mean(tf.reduce_mean(tf.square(x - x_hat), axis=[1, 2, 3]))

                # KL divergence
                kl = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
                )

                kl_w = self._kl_weight()
                total = recon + kl_w * kl

            grads = tape.gradient(total, self.trainable_weights)

            # CRITICAL: gradient clipping to prevent blow-ups
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]

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

    model = ConvVAE(encoder, decoder, beta=beta, kl_warmup_steps=kl_warmup_steps)
    return encoder, decoder, model

