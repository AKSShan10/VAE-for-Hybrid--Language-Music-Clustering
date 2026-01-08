# import tensorflow as tf
# from tensorflow.keras import layers, Model


# class Sampling(layers.Layer):
#     """Reparameterization trick: z = mu + eps * sigma."""
#     def call(self, inputs):
#         mu, logvar = inputs
#         eps = tf.random.normal(shape=tf.shape(mu))
#         return mu + tf.exp(0.5 * logvar) * eps


# def build_conditional_vae(input_shape_audio, input_shape_genre, latent_dim=32, beta=0.5):
#     """
#     Build the Conditional Variational Autoencoder (CVAE) model.
#     """

#     # Encoder
#     audio_input = layers.Input(shape=input_shape_audio, name="audio_input")
#     genre_input = layers.Input(shape=input_shape_genre, name="genre_input")

#     # Concatenate audio and genre
#     x = layers.Concatenate()([audio_input, genre_input])
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)

#     # Latent space (mu and logvar)
#     mu = layers.Dense(latent_dim, name="mu")(x)
#     logvar = layers.Dense(latent_dim, name="logvar")(x)

#     z = Sampling()([mu, logvar])

#     encoder = Model([audio_input, genre_input], [mu, logvar, z], name="encoder")

#     # Decoder
#     z_input = layers.Input(shape=(latent_dim,), name="z_input")
#     x = layers.Concatenate()([z_input, genre_input])
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)  # Assuming audio is a 1D vector

#     decoder = Model([z_input, genre_input], x, name="decoder")

#     # CVAE Model
#     cvae = Model([audio_input, genre_input], decoder(encoder([audio_input, genre_input])[2]), name="cvae")

#     # Define custom loss
#     def vae_loss(y_true, y_pred):
#         recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))  # SSE per sample
#         kl_loss = -0.5 * tf.reduce_mean(
#             tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
#         )
#         return recon_loss + beta * kl_loss

#     cvae.add_loss(vae_loss)
#     cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))

#     return encoder, decoder, cvae


# vae_conditional.py

# vae_conditional.py

# vae_conditional.py

# import tensorflow as tf
# from tensorflow.keras import layers, Model

# class Sampling(layers.Layer):
#     """Reparameterization trick: z = mu + eps * sigma."""
#     def call(self, inputs):
#         mu, logvar = inputs
#         eps = tf.random.normal(shape=tf.shape(mu))
#         return mu + tf.exp(0.5 * logvar) * eps

# def build_conditional_vae(input_shape_audio, input_shape_lyrics, latent_dim=32, beta=0.5):
#     """
#     Build the Conditional Variational Autoencoder (CVAE) model.
#     """

#     # Encoder
#     audio_input = layers.Input(shape=input_shape_audio, name="audio_input")  # shape: (None, 16384)
#     lyrics_input = layers.Input(shape=input_shape_lyrics, name="lyrics_input")  # shape: (None, 934)

#     # Flatten the audio input to make it 1D
#     x_audio = layers.Flatten()(audio_input)  # shape: (None, 16384)

#     # Concatenate audio and lyrics (genre) input
#     x = layers.Concatenate()([x_audio, lyrics_input])  # shape: (None, 16384 + 934 = 17318)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)

#     # Latent space (mu and logvar)
#     mu = layers.Dense(latent_dim, name="mu")(x)
#     logvar = layers.Dense(latent_dim, name="logvar")(x)

#     z = Sampling()([mu, logvar])

#     encoder = Model([audio_input, lyrics_input], [mu, logvar, z], name="encoder")

#     # Decoder
#     z_input = layers.Input(shape=(latent_dim,), name="z_input")
#     # Concatenate z with lyrics features (lyrics)
#     x = layers.Concatenate()([z_input, lyrics_input])  # shape: (None, latent_dim + 934)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)
#     # Output shape should match audio input's shape (e.g., spectrogram dimensions)
#     x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)  # Assuming audio is a 1D vector (shape: 16384)

#     decoder = Model([z_input, lyrics_input], x, name="decoder")

#     # CVAE Model
#     # mu, logvar, z = encoder([audio_input, lyrics_input])
#     # reconstructed = decoder([z, lyrics_input])

#     # cvae = Model([audio_input, lyrics_input], reconstructed, name="cvae")

#     # # Define custom loss
#     # def vae_loss(y_true, y_pred):
#     #     recon_loss = tf.reduce_mean(
#     #         tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
#     #     )  # Reconstruction loss (Mean Squared Error)
#     #     kl_loss = -0.5 * tf.reduce_mean(
#     #         tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
#     #     )  # KL Divergence
#     #     return recon_loss + beta * kl_loss

#     # cvae.add_loss(vae_loss)
#     # cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))

    #  My account
    # CVAE Model
    # mu, logvar, z = encoder([audio_input, lyrics_input])
    # reconstructed = decoder([z, lyrics_input])
    # cvae = Model([audio_input, lyrics_input], reconstructed, name="cvae")
    
    # # Add custom loss
    # recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(audio_input - reconstructed), axis=1))
    # kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))
    # cvae.add_loss(recon_loss + beta * kl_loss)
    
    # cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))


#     #  my gpt plus pro
#     # CVAE Model
#     mu, logvar, z = encoder([audio_input, lyrics_input])
#     reconstructed = decoder([z, lyrics_input])

#     cvae = Model([audio_input, lyrics_input], reconstructed, name="cvae")

#     # Define custom loss
#     def vae_loss(y_true, y_pred):
#         recon_loss = tf.reduce_mean(
#             tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
#         )  # Reconstruction loss (Mean Squared Error)
#         kl_loss = -0.5 * tf.reduce_mean(
#             tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
#         )  # KL Divergence
#         return recon_loss + beta * kl_loss

#     # The `add_loss` method works because Keras automatically adds the `y_true` and `y_pred`
#     # during training. Thus, no need to manually pass these arguments.
#     cvae.add_loss(vae_loss)
#     cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))


#     return encoder, decoder, cvae

# def build_conditional_vae(input_shape_audio, input_shape_lyrics, latent_dim=32, beta=0.5):
#     """
#     Build the Conditional Variational Autoencoder (CVAE) model.
#     """

#     # -------- Encoder --------
#     audio_input = layers.Input(shape=input_shape_audio, name="audio_input")  # shape: (None, 16384)
#     lyrics_input = layers.Input(shape=input_shape_lyrics, name="lyrics_input")  # shape: (None, 934)

#     # Flatten the audio input to make it 1D
#     x_audio = layers.Flatten()(audio_input)  # shape: (None, 16384)

#     # Concatenate audio and lyrics (genre) input
#     x = layers.Concatenate()([x_audio, lyrics_input])  # shape: (None, 16384 + 934 = 17318)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)

#     # Latent space (mu and logvar)
#     mu = layers.Dense(latent_dim, name="mu")(x)
#     logvar = layers.Dense(latent_dim, name="logvar")(x)

#     z = Sampling()([mu, logvar])

#     encoder = Model([audio_input, lyrics_input], [mu, logvar, z], name="encoder")

#     # -------- Decoder --------
#     z_input = layers.Input(shape=(latent_dim,), name="z_input")
#     # Concatenate z with lyrics features (lyrics)
#     x = layers.Concatenate()([z_input, lyrics_input])  # shape: (None, latent_dim + 934)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(256, activation="relu")(x)
#     # Output shape should match audio input's shape (e.g., spectrogram dimensions)
#     x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)

#     decoder = Model([z_input, lyrics_input], x, name="decoder")

#     # -------- CVAE Model --------
#     mu, logvar, z = encoder([audio_input, lyrics_input])
#     reconstructed = decoder([z, lyrics_input])

#     cvae = Model([audio_input, lyrics_input], reconstructed, name="cvae")

#     # -------- Loss --------
#     def vae_loss(y_true, y_pred):
#         """
#         Reconstruction + KL divergence loss for Conditional VAE.
#         Since this is unsupervised learning, y_true and y_pred are actually the same
#         and correspond to the input (audio features).
#         """
#         recon_loss = tf.reduce_mean(
#             tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
#         )  # Reconstruction loss (Mean Squared Error)

#         kl_loss = -0.5 * tf.reduce_mean(
#             tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
#         )  # KL Divergence

#         return recon_loss + beta * kl_loss

#     # The `add_loss` method works because Keras automatically adds the `y_true` and `y_pred`
#     # during training. Thus, no need to manually pass these arguments.
#     cvae.add_loss(vae_loss)
#     cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))

#     return encoder, decoder, cvae

# import tensorflow as tf
# from tensorflow.keras import layers, Model


# class Sampling(layers.Layer):
#     """Reparameterization trick: z = mu + eps * sigma."""
#     def call(self, inputs):
#         mu, logvar = inputs
#         eps = tf.random.normal(shape=tf.shape(mu))
#         return mu + tf.exp(0.5 * logvar) * eps


# # def build_conditional_vae(input_shape_audio, input_shape_lyrics, latent_dim=32, beta=0.5):
# #     """
# #     Build the Conditional Variational Autoencoder (CVAE) model.
# #     """

# #     # -------- Encoder --------
# #     audio_input = layers.Input(shape=input_shape_audio, name="audio_input")  # shape: (None, 16384)
# #     lyrics_input = layers.Input(shape=input_shape_lyrics, name="lyrics_input")  # shape: (None, 934)

# #     # Flatten the audio input to make it 1D
# #     x_audio = layers.Flatten()(audio_input)  # shape: (None, 16384)

# #     # Concatenate audio and lyrics (genre) input
# #     x = layers.Concatenate()([x_audio, lyrics_input])  # shape: (None, 16384 + 934 = 17318)
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.BatchNormalization()(x)
# #     x = layers.Dense(256, activation="relu")(x)

# #     # Latent space (mu and logvar)
# #     mu = layers.Dense(latent_dim, name="mu")(x)
# #     logvar = layers.Dense(latent_dim, name="logvar")(x)

# #     z = Sampling()([mu, logvar])

# #     encoder = Model([audio_input, lyrics_input], [mu, logvar, z], name="encoder")

# #     # -------- Decoder --------
# #     z_input = layers.Input(shape=(latent_dim,), name="z_input")
# #     # Concatenate z with lyrics features (lyrics)
# #     x = layers.Concatenate()([z_input, lyrics_input])  # shape: (None, latent_dim + 934)
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.BatchNormalization()(x)
# #     x = layers.Dense(256, activation="relu")(x)
# #     # Output shape should match audio input's shape (e.g., spectrogram dimensions)
# #     x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)

# #     decoder = Model([z_input, lyrics_input], x, name="decoder")

# #     # -------- CVAE Model --------
# #     mu, logvar, z = encoder([audio_input, lyrics_input])
# #     reconstructed = decoder([z, lyrics_input])

# #     cvae = Model([audio_input, lyrics_input], reconstructed, name="cvae")

# #     # -------- Loss --------
# #     def vae_loss(y_true, y_pred):
# #         """
# #         Reconstruction + KL divergence loss for Conditional VAE.
# #         Since this is unsupervised learning, y_true and y_pred are actually the same
# #         and correspond to the input (audio features).
# #         """
# #         recon_loss = tf.reduce_mean(
# #             tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
# #         )  # Reconstruction loss (Mean Squared Error)

# #         kl_loss = -0.5 * tf.reduce_mean(
# #             tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
# #         )  # KL Divergence

# #         return recon_loss + beta * kl_loss

# #     # The `add_loss` method works because Keras automatically adds the `y_true` and `y_pred`
# #     # during training. Thus, no need to manually pass these arguments.
# #     cvae.add_loss(vae_loss)
# #     cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))

# #     return encoder, decoder, cvae


# # def build_conditional_vae(input_shape_audio, input_shape_lyrics, latent_dim=32, beta=0.5):
# #     """
# #     Build the Conditional Variational Autoencoder (CVAE) model.
# #     """
# #     # -------- Encoder --------
# #     audio_input = layers.Input(shape=input_shape_audio, name="audio_input")
# #     lyrics_input = layers.Input(shape=input_shape_lyrics, name="lyrics_input")
    
# #     x_audio = layers.Flatten()(audio_input)
# #     x = layers.Concatenate()([x_audio, lyrics_input])
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.BatchNormalization()(x)
# #     x = layers.Dense(256, activation="relu")(x)
    
# #     mu = layers.Dense(latent_dim, name="mu")(x)
# #     logvar = layers.Dense(latent_dim, name="logvar")(x)
# #     z = Sampling()([mu, logvar])
    
# #     encoder = Model([audio_input, lyrics_input], [mu, logvar, z], name="encoder")
    
# #     # -------- Decoder --------
# #     z_input = layers.Input(shape=(latent_dim,), name="z_input")
# #     lyrics_input_decoder = layers.Input(shape=input_shape_lyrics, name="lyrics_input_decoder")
    
# #     x = layers.Concatenate()([z_input, lyrics_input_decoder])
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.BatchNormalization()(x)
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)
    
# #     decoder = Model([z_input, lyrics_input_decoder], x, name="decoder")
    
# #     # -------- Custom CVAE Model --------
# #     class CVAE(Model):
# #         def __init__(self, encoder, decoder, beta=0.5, **kwargs):
# #             super(CVAE, self).__init__(**kwargs)
# #             self.encoder = encoder
# #             self.decoder = decoder
# #             self.beta = beta
# #             self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
# #             self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
# #             self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
# #         @property
# #         def metrics(self):
# #             return [
# #                 self.total_loss_tracker,
# #                 self.recon_loss_tracker,
# #                 self.kl_loss_tracker,
# #             ]
        
# #         def call(self, inputs):
# #             audio_input, lyrics_input = inputs
# #             mu, logvar, z = self.encoder([audio_input, lyrics_input])
# #             reconstructed = self.decoder([z, lyrics_input])
# #             return reconstructed
        
# #         def train_step(self, data):
# #             # Handle different data formats
# #             if isinstance(data, tuple):
# #                 # If data is (x, y) or (x, y, sample_weight)
# #                 x = data[0]
# #             else:
# #                 # If data is just x
# #                 x = data
            
# #             # Unpack the inputs
# #             if isinstance(x, (list, tuple)):
# #                 audio_input, lyrics_input = x
# #             else:
# #                 # If x is a dict (unlikely but possible)
# #                 audio_input = x['audio_input']
# #                 lyrics_input = x['lyrics_input']
            
# #             with tf.GradientTape() as tape:
# #                 # Forward pass
# #                 mu, logvar, z = self.encoder([audio_input, lyrics_input])
# #                 reconstructed = self.decoder([z, lyrics_input])
                
# #                 # Compute losses
# #                 recon_loss = tf.reduce_mean(
# #                     tf.reduce_sum(tf.square(audio_input - reconstructed), axis=1)
# #                 )
# #                 kl_loss = -0.5 * tf.reduce_mean(
# #                     tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
# #                 )
# #                 total_loss = recon_loss + self.beta * kl_loss
            
# #             # Compute gradients and update weights
# #             grads = tape.gradient(total_loss, self.trainable_weights)
# #             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
# #             # Update metrics
# #             self.total_loss_tracker.update_state(total_loss)
# #             self.recon_loss_tracker.update_state(recon_loss)
# #             self.kl_loss_tracker.update_state(kl_loss)
            
# #             return {
# #                 "total_loss": self.total_loss_tracker.result(),
# #                 "recon_loss": self.recon_loss_tracker.result(),
# #                 "kl_loss": self.kl_loss_tracker.result(),
# #             }
        
# #         def test_step(self, data):
# #             # Handle different data formats (same as train_step)
# #             if isinstance(data, tuple):
# #                 x = data[0]
# #             else:
# #                 x = data
            
# #             if isinstance(x, (list, tuple)):
# #                 audio_input, lyrics_input = x
# #             else:
# #                 audio_input = x['audio_input']
# #                 lyrics_input = x['lyrics_input']
            
# #             # Forward pass
# #             mu, logvar, z = self.encoder([audio_input, lyrics_input])
# #             reconstructed = self.decoder([z, lyrics_input])
            
# #             # Compute losses
# #             recon_loss = tf.reduce_mean(
# #                 tf.reduce_sum(tf.square(audio_input - reconstructed), axis=1)
# #             )
# #             kl_loss = -0.5 * tf.reduce_mean(
# #                 tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
# #             )
# #             total_loss = recon_loss + self.beta * kl_loss
            
# #             # Update metrics
# #             self.total_loss_tracker.update_state(total_loss)
# #             self.recon_loss_tracker.update_state(recon_loss)
# #             self.kl_loss_tracker.update_state(kl_loss)
            
# #             return {
# #                 "total_loss": self.total_loss_tracker.result(),
# #                 "recon_loss": self.recon_loss_tracker.result(),
# #                 "kl_loss": self.kl_loss_tracker.result(),
# #             }
    
# #     cvae = CVAE(encoder, decoder, beta=beta, name="cvae")
    
# #     return encoder, decoder, cvae



import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + eps * sigma."""
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

def build_conditional_vae(input_shape_audio, input_shape_lyrics, latent_dim=32, beta=0.5):
    """
    Build the Conditional Variational Autoencoder (CVAE) model.
    """
    # -------- Encoder --------
    audio_input = layers.Input(shape=input_shape_audio, name="audio_input")
    lyrics_input = layers.Input(shape=input_shape_lyrics, name="lyrics_input")
    
    x_audio = layers.Flatten()(audio_input)
    x = layers.Concatenate()([x_audio, lyrics_input])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    mu = layers.Dense(latent_dim, name="mu")(x)
    logvar = layers.Dense(latent_dim, name="logvar")(x)
    # Clip logvar to prevent numerical instability
    logvar = layers.Lambda(lambda x: tf.clip_by_value(x, -10, 10))(logvar)
    
    z = Sampling()([mu, logvar])
    
    encoder = Model([audio_input, lyrics_input], [mu, logvar, z], name="encoder")
    
    # -------- Decoder --------
    z_input = layers.Input(shape=(latent_dim,), name="z_input")
    lyrics_input_decoder = layers.Input(shape=input_shape_lyrics, name="lyrics_input_decoder")
    
    x = layers.Concatenate()([z_input, lyrics_input_decoder])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(input_shape_audio[0], activation="sigmoid")(x)
    
    decoder = Model([z_input, lyrics_input_decoder], x, name="decoder")
    
    # -------- Custom CVAE Model --------
    class CVAE(Model):
        def __init__(self, encoder, decoder, beta=0.5, **kwargs):
            super(CVAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.beta = beta
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker,
            ]
        
        def call(self, inputs):
            audio_input, lyrics_input = inputs
            mu, logvar, z = self.encoder([audio_input, lyrics_input])
            reconstructed = self.decoder([z, lyrics_input])
            return reconstructed
        
        def train_step(self, data):
            # Handle different data formats
            if isinstance(data, tuple):
                x = data[0]
            else:
                x = data
            
            if isinstance(x, (list, tuple)):
                audio_input, lyrics_input = x
            else:
                audio_input = x['audio_input']
                lyrics_input = x['lyrics_input']
            
            with tf.GradientTape() as tape:
                # Forward pass
                mu, logvar, z = self.encoder([audio_input, lyrics_input])
                reconstructed = self.decoder([z, lyrics_input])
                
                # Compute reconstruction loss (MSE)
                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(audio_input - reconstructed), axis=1)
                )
                
                # Compute KL divergence with numerical stability
                kl_loss = -0.5 * tf.reduce_sum(
                    1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1
                )
                kl_loss = tf.reduce_mean(kl_loss)
                
                # Total loss
                total_loss = recon_loss + self.beta * kl_loss
                
                # Check for NaN
                total_loss = tf.debugging.check_numerics(total_loss, "Loss is NaN or Inf")
            
            # Compute gradients and update weights
            grads = tape.gradient(total_loss, self.trainable_weights)
            
            # Clip gradients to prevent explosion
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
            
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            # Update metrics
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "total_loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
        def test_step(self, data):
            if isinstance(data, tuple):
                x = data[0]
            else:
                x = data
            
            if isinstance(x, (list, tuple)):
                audio_input, lyrics_input = x
            else:
                audio_input = x['audio_input']
                lyrics_input = x['lyrics_input']
            
            mu, logvar, z = self.encoder([audio_input, lyrics_input])
            reconstructed = self.decoder([z, lyrics_input])
            
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(audio_input - reconstructed), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )
            total_loss = recon_loss + self.beta * kl_loss
            
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "total_loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
    
    cvae = CVAE(encoder, decoder, beta=beta, name="cvae")
    
    return encoder, decoder, cvae








