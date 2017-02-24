from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np


class GAN:
    
    OPTIMIZER = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    
    def __init__(self, generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER):
        self.generator = generator
        self.discriminator = discriminator
        self.d_optim = d_optim
        self.g_optim = g_optim
        
    def _create_adversarial_model(self):
        self.discriminator_on_generator_ = Sequential()
        self.discriminator_on_generator_.add(self.generator)
        self.discriminator_on_generator_.add(self.discriminator)
        
    def _sample_noise(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, self.generator.get_input_shape_at(0)[1]))

    @staticmethod    
    def _sample_data(X, batch_size, batch_index):
        return X[batch_index * batch_size:(batch_index + 1) * batch_size]
        
    def _train_discriminator(self, X, batch_size, batch_index):
        X_batch_data = GAN._sample_data(X, batch_size, batch_index)
        X_batch_generated = self.generator.predict(self._sample_noise(batch_size))
        X_discriminator = np.concatenate((X_batch_data, X_batch_generated))
        y_discriminator = [1] * batch_size + [0] * batch_size
        d_loss = self.discriminator.train_on_batch(X_discriminator, y_discriminator)
        return d_loss
        
    def _train_discriminator_on_generator(self, batch_size):
        g_loss = self.discriminator_on_generator_.train_on_batch(self._sample_noise(batch_size), [1] * batch_size)
        return g_loss
    
    def _compile_networks(self):
        self.discriminator.trainable = False
        self.discriminator_on_generator_.compile(loss='binary_crossentropy', optimizer=self.g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optim)
    
    def train(self, X, nb_epoch, batch_size, verbose=2):
        num_batches = int(X.shape[0] / batch_size)
        self._create_adversarial_model()
        self._compile_networks()
        for epoch in range(nb_epoch):
            for batch_index in range(num_batches):
                d_loss = self._train_discriminator(X, batch_size, batch_index)
                g_loss = self._train_discriminator_on_generator(batch_size)
                if verbose == 2:
                    print('Epoch %d, Batch %d, d_loss: %f, g_loss: %f' % (epoch, batch_index, d_loss, g_loss))    
            if verbose == 1:
                print('Epoch %d, d_loss: %f, g_loss: %f' % (epoch, d_loss, g_loss))
    
    def generate(self, n_samples):
        return self.generator.predict(self._sample_noise(n_samples))