
"""
This module contains the classes for Generative 
Adversarial Networks (GAN) and Conditional Generative 
Adversarial Networks (CGAN).
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt


OPTIMIZER = tf.train.AdamOptimizer()

def bind_columns(tensor1, tensor2):
    """Column binds the tensors if the second tensor exists, 
    else returns the first tensor."""
    if tensor2 is None:
        return tensor1
    return tf.concat(axis=1, values=[tensor1, tensor2])

def initialize_model_parameters(n_input_units, n_output_units, initialization_choice):
    shape = [n_input_units, n_output_units] if n_output_units is not None else [n_input_units]
    initialization_types = {'xavier': tf.random_normal(shape=shape, stddev=1. / tf.sqrt(n_input_units / 2.)), 
                            'normal': tf.random_normal(shape=shape), 
                            'zeros': tf.zeros(shape=shape)}
    return initialization_types[initialization_choice]

def initialize_model(model_layers, input_layer_correction, weights_initialization_choice, bias_initialization_choice):
    """Initializes variables for the model parameters and 
    a placeholder for the input data."""
    model_parameters = {}
    for layer_index in range(len(model_layers) - 1):
        model_parameters['W' + str(layer_index)] = tf.Variable(initialize_model_parameters(model_layers[layer_index][0], model_layers[layer_index + 1][0], weights_initialization_choice))
        model_parameters['b' + str(layer_index)] = tf.Variable(initialize_model_parameters(model_layers[layer_index + 1][0], None, bias_initialization_choice))
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

def sample_y(n_samples, n_y_features, class_label):
    """Returns a matrix of (n_samples, n_y_features) shape using 
    one-hot encoding for the class label. """
    if n_y_features > 2:
        y = np.zeros(shape=[n_samples, n_y_features], dtype='float32')
        y[:, class_label] = 1.
    else:
        y = np.zeros([n_samples, 1], dtype='float32') if class_label == 0 else np.ones([n_samples, 1], dtype='float32')
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

def split_data(X, y, validation_split):
    n_samples = X.shape[0]
    permuted_indices = np.random.permutation(range(n_samples))
    n_validation_indices = np.floor(validation_split * n_samples).astype(int)
    X_val, y_val = X[:n_validation_indices], y[:n_validation_indices] if y is not None else None
    X_train, y_train = X[n_validation_indices:], y[n_validation_indices:] if y is not None else None
    return X_train, y_train, X_val, y_val

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

    def __init__(self, n_Z_features,
                       discriminator_hidden_layers, 
                       generator_hidden_layers, 
                       discriminator_optimizer=OPTIMIZER,  
                       discriminator_weights_initilization_choice='xavier',
                       discriminator_bias_initilization_choice='zeros',
                       generator_optimizer=OPTIMIZER, 
                       generator_weights_initilization_choice='xavier',
                       generator_bias_initilization_choice='zeros'):
        self.n_Z_features = n_Z_features
        self.discriminator_hidden__layers = discriminator_hidden_layers
        self.generator_hidden_layers = generator_hidden_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_weights_initilization_choice = discriminator_weights_initilization_choice
        self.discriminator_bias_initilization_choice = discriminator_bias_initilization_choice
        self.generator_weights_initilization_choice = generator_weights_initilization_choice
        self.generator_bias_initilization_choice = generator_bias_initilization_choice

    def _initialize_training_parameters(self, X, y, batch_size):
        """Private method that initializes the GAN training parameters."""
        self.n_X_features = X.shape[1]
        self.n_y_features = y.shape[1] if y is not None else 0

        self.discriminator_layers = [(self.n_X_features + self.n_y_features, None)] + self.discriminator_hidden__layers + [(1, None)]
        self.generator_layers = [(self.n_Z_features + self.n_y_features, None)] + self.discriminator_hidden__layers + [(self.n_X_features, None)]

        self.y_placeholder = tf.placeholder(tf.float32, [None, self.n_y_features]) if y is not None else None
        self.X_placeholder, self.discriminator_parameters = initialize_model(self.discriminator_layers, self.n_y_features, self.discriminator_weights_initilization_choice, self.discriminator_bias_initilization_choice)
        self.Z_placeholder, self.generator_parameters = initialize_model(self.generator_layers, self.n_y_features, self.generator_weights_initilization_choice, self.generator_bias_initilization_choice)
        
        generator_logit = output_logits_tensor(bind_columns(self.Z_placeholder, self.y_placeholder), self.generator_layers, self.generator_parameters)
        discriminator_logit_real = output_logits_tensor(bind_columns(self.X_placeholder, self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        discriminator_logit_generated = output_logits_tensor(bind_columns(tf.nn.sigmoid(generator_logit), self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        
        self.discriminator_loss_mixed_data = return_loss(discriminator_logit_real, True) + return_loss(discriminator_logit_generated, False)
        self.discriminator_loss_generated_data = return_loss(discriminator_logit_generated, True)

        self.discriminator_optimization = define_optimization(self.discriminator_optimizer, self.discriminator_loss_mixed_data, self.discriminator_parameters)
        self.generator_optimization = define_optimization(self.generator_optimizer, self.discriminator_loss_generated_data, self.generator_parameters)

        self.discriminator_placeholders = [placeholder for placeholder in [self.X_placeholder, self.Z_placeholder, self.y_placeholder] if placeholder is not None]
        self.generator_placeholders = [placeholder for placeholder in [self.Z_placeholder, self.y_placeholder] if placeholder is not None]

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def _return_epoch_loss_value(self, X, y, batch_size, session, model_optimization, model_loss, placeholders, shuffle):
        """Private method that returns the loss function value for an 
        epoch of training."""
        n_samples = X.shape[0]
        if shuffle:
            X_epoch, y_epoch = shuffle_data(X, y)
        else:
            X_epoch, y_epoch = X, y
        mini_batch_indices = mini_batch_indices_generator(n_samples, batch_size)
        n_batches = n_samples // batch_size
        total_epoch_loss_value = 0
        for batch_index in range(n_batches):
            mb_indices = next(mini_batch_indices)
            adjusted_batch_size = mb_indices[1] - mb_indices[0]
            X_batch, y_batch = create_mini_batch_data(X_epoch, y_epoch, mb_indices)
            feed_dict = {self.X_placeholder: X_batch, self.Z_placeholder: sample_Z(adjusted_batch_size, self.n_Z_features), self.y_placeholder: y_batch}
            feed_dict = {placeholder: data for placeholder, data in feed_dict.items() if placeholder in placeholders}
            if model_optimization is not None:
                _, mb_loss_value = session.run([model_optimization, model_loss], feed_dict=feed_dict)
            else:
                mb_loss_value = session.run(model_loss, feed_dict=feed_dict)
            total_epoch_loss_value += mb_loss_value * adjusted_batch_size
        return total_epoch_loss_value / n_samples


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

    def train(self, X, nb_epoch, batch_size, validation_split=0.0, discriminator_steps=1):
        """Trains the GAN with X as the input data for nb_epoch number of epochs, 
        batch_size the size of the mini batch and discriminator_steps as the number 
        of discriminator gradient updates for each generator gradient update."""
        X_train, y_train, X_val, y_val = split_data(X, None, validation_split)
        super()._initialize_training_parameters(X_train, y_train, batch_size)
        for epoch in range(nb_epoch):
            for _ in range(discriminator_steps):
                discriminator_loss_mixed_training_data = self._return_epoch_loss_value(X_train, y_train, batch_size, self.sess, self.discriminator_optimization, self.discriminator_loss_mixed_data, self.discriminator_placeholders, True)
            discriminator_loss_generated_data = self._return_epoch_loss_value(X_train, y_train, batch_size, self.sess, self.generator_optimization, self.discriminator_loss_generated_data, self.generator_placeholders, True)
            if X_val.size > 0:
                discriminator_loss_mixed_validation_data = self._return_epoch_loss_value(X_val, y_val, batch_size, self.sess, None, self.discriminator_loss_mixed_data, self.discriminator_placeholders, False)

            print('Epoch: {}\nDiscriminator loss on mixed training data: {}\nDiscriminator loss on mixed validation data: {}\nDiscriminator loss on generated data: {}\n'.format(epoch, discriminator_loss_mixed_training_data, discriminator_loss_mixed_validation_data, discriminator_loss_generated_data))
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
        be equal to (X.shape[1] + n_y_features, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (1, None), where X is the input data and n_y_features is 
        the number of y matrix features.
    generator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding layer. The number 
        of neurons and the activation function for the first layer should 
        be equal to (Z.shape[1] + n_y_features, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (X.shape[1], None), where X is the input data and n_y_features 
        is the number of y matrix features.
    discriminator_optimizer: TensorFlow optimizer
        The optimizer for the discriminator.
    generator_optimizer: TensorFlow optimizer
        The optimizer for the generator.
    """

    def train(self, X, y, nb_epoch, batch_size, validation_split=0.0, discriminator_steps=1):
        """Trains the Conditional GAN with X as the input data, y the one-hot
        encoded class labels for nb_epoch number of epochs, batch_size the size 
        of the mini batch, discriminator_steps as the number of discriminator 
        gradient updates for each generator gradient update."""
        X_train, y_train, X_val, y_val = split_data(X, y, validation_split)
        super()._initialize_training_parameters(X_train, y_train, batch_size)
        for epoch in range(nb_epoch):
            for _ in range(discriminator_steps):
                discriminator_loss_mixed_training_data = self._return_epoch_loss_value(X_train, y_train, batch_size, self.sess, self.discriminator_optimization, self.discriminator_loss_mixed_data, self.discriminator_placeholders, True)
            discriminator_loss_generated_data = self._return_epoch_loss_value(X_train, y_train, batch_size, self.sess, self.generator_optimization, self.discriminator_loss_generated_data, self.generator_placeholders, True)
            if X_val.size > 0:
                discriminator_loss_mixed_validation_data = self._return_epoch_loss_value(X_val, y_val, batch_size, self.sess, None, self.discriminator_loss_mixed_data, self.discriminator_placeholders, False)

            print('Epoch: {}\nDiscriminator loss on mixed training data: {}\nDiscriminator loss on mixed validation data: {}\nDiscriminator loss on generated data: {}\n'.format(epoch, discriminator_loss_mixed_training_data, discriminator_loss_mixed_validation_data, discriminator_loss_generated_data))
        return self

    def generate_samples(self, n_samples, class_label):
        """Generates n_samples number from the generator 
        conditioned on the class_label."""
        input_tensor = np.concatenate([sample_Z(n_samples, self.n_Z_features), sample_y(n_samples, self.n_y_features, class_label)], axis=1)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples
