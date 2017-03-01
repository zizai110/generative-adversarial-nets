from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


class GAN:
    
    OPTIMIZER = Adam()
    
    def __init__(self, generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER):
        self.generator = generator
        self.discriminator = discriminator
        self.d_optim = d_optim
        self.g_optim = g_optim
        
    def _prepare_adversarial_models(self):
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optim)

        self.full_model = Sequential()
        self.full_model.add(self.generator)
        self.full_model.add(self.discriminator)
        self.discriminator.trainable = False
        self.full_model.compile(loss='binary_crossentropy', optimizer=self.g_optim)
        
    def _sample_generator_data(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, self.generator.get_input_shape_at(0)[1]))

    def _sample_discriminator_data(self, X, batch_size, batch_index):
        X_batch = X[batch_index * batch_size:(batch_index + 1) * batch_size]
        Z_batch = self.generator.predict(self._sample_noise(batch_size))
        X_discriminator = np.concatenate((X_batch, Z_batch))
        return X_discriminator
        
    def train(self, X, nb_epoch, batch_size, verbose=2):
        num_batches = int(X.shape[0] / batch_size)
        self._prepare_adversarial_models()
        for epoch in range(nb_epoch):
            for batch_index in range(num_batches):
                d_loss = self.discriminator.train_on_batch(self._sample_discriminator_data(X, batch_size, batch_index), [1] * batch_size + [0] * batch_size)
                g_loss = self.full_model.train_on_batch(self._sample_generator_data(batch_size), [1] * batch_size)
                if verbose == 2:
                    print('Epoch %d, Batch %d, d_loss: %f, g_loss: %f' % (epoch, batch_index, d_loss, g_loss))    
            if verbose == 1:
                print('Epoch %d, d_loss: %f, g_loss: %f' % (epoch, d_loss, g_loss))
    
    def generate(self, n_samples):
        return self.generator.predict(self._sample_generator_data(n_samples))