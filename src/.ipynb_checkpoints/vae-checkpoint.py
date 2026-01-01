import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_recon = self.decoder(z, training=True)

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_vae(input_dim: int, latent_dim: int = 2, hidden_dim: int = 64, beta: float = 1.0):
    # Encoder
    x_in = keras.Input(shape=(input_dim,), name="x_in")
    h = layers.Dense(hidden_dim, activation="relu")(x_in)
    h = layers.Dense(hidden_dim, activation="relu")(h)

    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)

    def sampling(args):
        zm, zv = args
        eps = tf.random.normal(shape=tf.shape(zm))
        return zm + tf.exp(0.5 * zv) * eps

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = keras.Model(x_in, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    z_in = keras.Input(shape=(latent_dim,), name="z_in")
    h2 = layers.Dense(hidden_dim, activation="relu")(z_in)
    h2 = layers.Dense(hidden_dim, activation="relu")(h2)
    x_out = layers.Dense(input_dim, activation="linear", name="x_out")(h2)
    decoder = keras.Model(z_in, x_out, name="decoder")

    vae = VAE(encoder, decoder, beta=beta, name="vae")
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    return vae, encoder, decoder
