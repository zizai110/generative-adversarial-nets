
"""
This module contains the classes for Generative 
Adversarial Networks (GAN) and Conditional Generative 
Adversarial Networks (CGAN).
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
import tensorflow as tf


OPTIMIZER = tf.train.AdamOptimizer()

def bind_columns(tensor1, tensor2):
    """Column binds the tensors if the second tensor exists, 
    else returns the first tensor."""
    if tensor2 is None:
        return tensor1
    return tf.concat(axis=1, values=[tensor1, tensor2])

def initialize_model(model_layers, input_layer_correction):
    """Initializes variables for the model parameters and 
    a placeholder for the input data."""
    model_parameters = {}
    for layer_index in range(len(model_layers) - 1):
        model_parameters['W' + str(layer_index)] = tf.Variable(tf.random_normal([model_layers[layer_index][0], model_layers[layer_index + 1][0]]))
        model_parameters['b' + str(layer_index)] = tf.Variable(tf.zeros([model_layers[layer_index + 1][0]]))
    input_data_placeholder = tf.placeholder(tf.float32, shape=[None, model_layers[0][0] - input_layer_correction])
    return input_data_placeholder, model_parameters
    
def output_logits_tensor(input_tensor, model_layers, model_parameters):
    """Returns the output logits of a model given its parameters and 
    an input tensor."""
    output_tensor = input_tensor
    for layer_index in range(len(model_layers) - 1):
        logit_tensor = tf.matmul(output_tensor, model_parameters['W' + str(layer_index)]) + model_parameters['b' + str(layer_index)]
        activation_function = model_layers[layer_index + 1][1]
        if activation_function is not None:
            output_tensor = activation_function(logit_tensor)
        else:
            output_tensor = logit_tensor
    return output_tensor

def sample_Z(n_samples, n_features):
    """Samples the elements of a (n_samples, n_features) shape 
    matrix from a uniform distribution in the [-1, 1] interval."""
    return np.random.uniform(-1., 1., size=[n_samples, n_features]).astype(np.float32)

def sample_y(n_samples, n_classes, class_label):
    """Returns a matrix of (n_samples, n_classes) shape using 
    one-hot encoding for the class label. """
    y = np.zeros(shape=[n_samples, n_classes]).astype(np.float32)
    y[:, class_label] = 1.
    return y
    
def return_loss(logits, positive_class_labels=True):
    """Returns the loss function of the discriminator or generator 
    for  positive or negative class labels."""
    if positive_class_labels:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))
    return loss

def define_optimization(optimizer, loss, model_parameters):
    """Defines the optimization problem for a given optimizer, 
    loss function and model parameters."""
    return optimizer.minimize(loss, var_list=list(model_parameters.values()))

def shuffle_data(X, y):
    """Shuffle the data."""
    epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
    X_epoch = X[epoch_shuffled_indices]
    y_epoch = y[epoch_shuffled_indices] if y is not None else None
    return X_epoch, y_epoch

def create_mini_batch_data(X, y, mini_batch_indices):
    """Return a mini batch of the data."""
    X_batch = X[slice(*mini_batch_indices)]
    y_batch = y[slice(*mini_batch_indices)] if y is not None else None
    return X_batch, y_batch
                        
def mini_batch_indices_generator(n_samples, batch_size):
    """A generator of the mini batch indices based on the 
    number of samples and the batch size."""
    start_index = 0
    end_index = batch_size
    while start_index < n_samples:
        yield start_index, end_index
        start_index += batch_size
        if end_index + batch_size <= n_samples:
            end_index += batch_size
        else:
            end_index += n_samples % batch_size


class BaseGAN:
    """Base class for GANs and CGANs."""  

    def __init__(self, discriminator_layers, generator_layers, discriminator_optimizer=OPTIMIZER, generator_optimizer=OPTIMIZER):
        self.discriminator_layers = discriminator_layers
        self.generator_layers = generator_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def _initialize_training_parameters(self, X, y, batch_size):
        """Private method that initializes the GAN training parameters."""
        self.n_classes = y.shape[1] if y is not None else 0
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.n_classes]) if y is not None else None
        self.X_placeholder, self.discriminator_parameters = initialize_model(self.discriminator_layers, self.n_classes)
        self.Z_placeholder, self.generator_parameters = initialize_model(self.generator_layers, self.n_classes)
        
        generator_logit = output_logits_tensor(bind_columns(self.Z_placeholder, self.y_placeholder), self.generator_layers, self.generator_parameters)
        discriminator_logit_real = output_logits_tensor(bind_columns(self.X_placeholder, self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        discriminator_logit_generated = output_logits_tensor(bind_columns(tf.nn.sigmoid(generator_logit), self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        
        self.discriminator_loss = return_loss(discriminator_logit_real, True) + return_loss(discriminator_logit_generated, False)
        self.generator_loss = return_loss(discriminator_logit_generated, True)

        self.discriminator_optimization = define_optimization(self.discriminator_optimizer, self.discriminator_loss, self.discriminator_parameters)
        self.generator_optimization = define_optimization(self.generator_optimizer, self.generator_loss, self.generator_parameters)

        self.n_X_samples = X.shape[0]
        self.n_batches = self.n_X_samples // batch_size
        self.n_Z_features = self.generator_layers[0][0] - self.n_classes

        self.discriminator_placeholders = [placeholder for placeholder in [self.X_placeholder, self.Z_placeholder, self.y_placeholder] if placeholder is not None]
        self.generator_placeholders = [placeholder for placeholder in [self.Z_placeholder, self.y_placeholder] if placeholder is not None]

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def _return_epoch_loss_value(self, X, y, batch_size, session, model_optimization, model_loss, placeholders):
        """Private method that returns the loss function value for an 
        epoch of training."""
        X_epoch, y_epoch = shuffle_data(X, y)
        mini_batch_indices = mini_batch_indices_generator(self.n_X_samples, batch_size)
        for batch_index in range(self.n_batches):
            X_batch, y_batch = create_mini_batch_data(X, y, next(mini_batch_indices))
            feed_dict = {self.X_placeholder: X_batch, self.Z_placeholder: sample_Z(batch_size, self.n_Z_features), self.y_placeholder: y_batch}
            feed_dict = {placeholder: data for placeholder, data in feed_dict.items() if placeholder in placeholders}
            _, loss_value = session.run([model_optimization, model_loss], feed_dict=feed_dict)
        return loss_value

    def _train_gan(self, X, y, nb_epoch, batch_size, discriminator_steps, verbose, session):
        """Private method that trains the GAN."""
        for epoch in range(nb_epoch):
            for _ in range(discriminator_steps):
                discriminator_loss_value = self._return_epoch_loss_value(X, y, batch_size, session, self.discriminator_optimization, self.discriminator_loss, self.discriminator_placeholders)
            generator_loss_value = self._return_epoch_loss_value(X, y, batch_size, session, self.generator_optimization, self.generator_loss, self.generator_placeholders)

            print('Epoch: {}, discriminator loss: {}, generator loss: {}'.format(epoch, discriminator_loss_value, generator_loss_value))


class GAN(BaseGAN):
    """
    Parameters
    ----------
    discriminator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the discriminator's corresponding layer. The 
        number of neurons and the activation function for the first 
        layer should be equal to (X.shape[1], None) while the number 
        of neurons and the activation function for the last layer should 
        be equal to (1, None), where X is the input data.
    generator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding layer. The 
        number of neurons and the activation function for the first 
        layer should be equal (Z.shape[1], None)while the number of neurons 
        and the activation function for the last layer should be equal to 
        (X.shape[1], None), where X is the input data.
    discriminator_optimizer: TensorFlow optimizer
        The optimizer for the discriminator.
    generator_optimizer: TensorFlow optimizer
        The optimizer for the generator.
    """

    def train(self, X, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        """Trains the GAN with X as the input data for nb_epoch number of epochs, 
        batch_size the size of the mini batch, discriminator_steps as the number 
        of discriminator gradient updates for each generator gradient update and 
        verbose the level of verbosity."""
        super()._initialize_training_parameters(X, None, batch_size)
        super()._train_gan(X, None, nb_epoch, batch_size, discriminator_steps, verbose, self.sess)
        return self

    def generate_samples(self, n_samples):
        """Generates n_samples from the generator."""
        input_tensor = sample_Z(n_samples, self.n_Z_features)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples

class CGAN(BaseGAN):
    """
    Parameters
    ----------
    discriminator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the discriminator's corresponding layer. The number 
        of neurons and the activation function for the first layer should 
        be equal to (X.shape[1] + n_classes, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (1, None), where X is the input data and n_classes is 
        the number of class labels in y.
    generator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding layer. The number 
        of neurons and the activation function for the first layer should 
        be equal to (Z.shape[1] + n_classes, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (X.shape[1], None), where X is the input data and n_classes 
        is the number of class labels in y.
    discriminator_optimizer: TensorFlow optimizer
        The optimizer for the discriminator.
    generator_optimizer: TensorFlow optimizer
        The optimizer for the generator.
    """

    def train(self, X, y, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        """Trains the Conditional GAN with X as the input data, y the class labels 
        for nb_epoch number of epochs, batch_size the size of the mini batch, 
        discriminator_steps as the number of discriminator gradient updates for 
        each generator gradient update and verbose the level of verbosity."""
        super()._initialize_training_parameters(X, y, batch_size)
        super()._train_gan(X, y, nb_epoch, batch_size, discriminator_steps, verbose, self.sess)
        return self

    def generate_samples(self, n_samples, class_label):
        """Generates n_samples number from the generator 
        conditioned on the class_label."""
        input_tensor = np.concatenate([sample_Z(n_samples, self.n_Z_features), sample_y(n_samples, self.n_classes, class_label)], axis=1)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples
