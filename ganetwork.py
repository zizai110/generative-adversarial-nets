from abc import ABCMeta, abstractmethod
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


OPTIMIZER = Adam()

class BaseGAN(metaclass=ABCMeta):

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

    @abstractmethod    
    def _sample_generator_data(self):
        pass

    @abstractmethod    
    def _sample_discriminator_data(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def generate(self):
        pass


class GAN(BaseGAN):

    def __init__(self, generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER):
        super().__init__(generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER)
        self.n_Z_features = self.generator.get_input_shape_at(0)[1]

    def _sample_generator_data(self, n_samples):
        return np.random.uniform(-1, 1, (n_samples, self.n_Z_features))

    def _sample_discriminator_data(self, X, n_samples):
        X_generated = self.generator.predict(self._sample_generator_data(n_samples))
        return np.concatenate([X, X_generated], axis=0)
        
    def train(self, X, nb_epoch, batch_size, verbose=2):
        num_batches = int(X.shape[0] / batch_size)
        self._prepare_adversarial_models()
        for epoch in range(nb_epoch):
            for batch_index in range(num_batches):
                X_batch = X[batch_index * batch_size:(batch_index + 1) * batch_size]
                d_loss = self.discriminator.train_on_batch(self._sample_discriminator_data(X_batch, batch_size), [1] * batch_size + [0] * batch_size)
                g_loss = self.full_model.train_on_batch(self._sample_generator_data(batch_size), [1] * batch_size)
                if verbose == 2:
                    print('Epoch %d, Batch %d, d_loss: %f, g_loss: %f' % (epoch, batch_index, d_loss, g_loss))    
            if verbose == 1:
                print('Epoch %d, d_loss: %f, g_loss: %f' % (epoch, d_loss, g_loss))
    
    def generate(self, n_samples):
        return self.generator.predict(self._sample_generator_data(n_samples))


class CGAN(BaseGAN):

    def __init__(self, generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER):
        super().__init__(generator, discriminator, d_optim=OPTIMIZER, g_optim=OPTIMIZER)
        self.n_generator_features = self.generator.get_input_shape_at(0)[1]

    def _sample_generator_data(self, y, n_samples):
        return np.concatenate([np.random.uniform(-1, 1, (n_samples, self.n_generator_features - y.shape[1])), y], axis=1)

    def _sample_discriminator_data(self, X, y, n_samples):
        X_y = np.concatenate([X, y], axis=1)
        X_generated_y = self.generator.predict(self._sample_generator_data(y, n_samples))
        return np.concatenate([X_y, X_generated_y], axis=0)
        
    def train(self, X, y, nb_epoch, batch_size, verbose=2):
        self.n_classes = y.shape[1]
        num_batches = int(X.shape[0] / batch_size)
        self._prepare_adversarial_models()
        for epoch in range(nb_epoch):
            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                X_batch, y_batch = X[start_index:end_index], y[start_index:end_index]
                d_loss = self.discriminator.train_on_batch(self._sample_discriminator_data(X_batch, y_batch, batch_size), [1] * batch_size + [0] * batch_size)
                g_loss = self.full_model.train_on_batch(np.concatenate([self._sample_generator_data(y_batch, batch_size), y_batch], axis=1), [1] * batch_size)
                if verbose == 2:
                    print('Epoch %d, Batch %d, d_loss: %f, g_loss: %f' % (epoch, batch_index, d_loss, g_loss))    
            if verbose == 1:
                print('Epoch %d, d_loss: %f, g_loss: %f' % (epoch, d_loss, g_loss))
    
    def generate(self, n_samples, class_label):
        y = np.zeros([n_samples, self.n_classes])
        y[:, class_label] = 1
        return self.generator.predict(self._sample_generator_data(y, n_samples))