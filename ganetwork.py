import numpy as np
import tensorflow as tf


OPTIMIZER = tf.train.AdamOptimizer()

def _initialize_model(model_layers, input_layer_correction):
    model_parameters = {}
    for layer_index in range(len(model_layers) - 1):
        model_parameters['W' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index][0], model_layers[layer_index + 1][0]]))
        model_parameters['b' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index + 1][0]]))
    input_data_placeholder = tf.placeholder(tf.float32, shape=[None, model_layers[0][0] - input_layer_correction])
    return input_data_placeholder, model_parameters
    
def _output_logit_tensor(input_tensor, model_layers, model_parameters):
    output_tensor = input_tensor
    for layer_index in range(len(model_layers) - 1):
        logit_tensor = tf.matmul(output_tensor, model_parameters['W' + str(layer_index)]) + model_parameters['b' + str(layer_index)]
        activation_function = model_layers[layer_index + 1][1]
        if activation_function is not None:
            output_tensor = activation_function(logit_tensor)
        else:
            output_tensor = logit_tensor
    return output_tensor

def _sample_Z(n_samples, n_features):
    return np.random.uniform(-1., 1., size=[n_samples, n_features]).astype(np.float32)

def _sample_y(n_samples, n_classes, class_label):
    y = np.zeros(shape=[n_samples, n_classes]).astype(np.float32)
    y[:, class_label] = 1.
    return y
    
def _return_loss(logits, positive_class_labels=True):
    if positive_class_labels:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))
    else:
        loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))
    return loss

def _update_model_parameters(optimizer, loss, model_parameters):
    return optimizer.minimize(loss, var_list=list(model_parameters.values()))

def mini_batch_indices_generator(n_samples, batch_size):
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

    def __init__(self, discriminator_layers, generator_layers, discriminator_optimizer=OPTIMIZER, generator_optimizer=OPTIMIZER):
        self.discriminator_layers = discriminator_layers
        self.generator_layers = generator_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def _prepare_training(self, X, y, batch_size):
        if y is None:
            self.n_classes = 0
            self.y_placeholder = None
        else:
            self.n_classes = y.shape[1]
            self.y_placeholder = tf.placeholder(tf.float32, [None, self.n_classes])
        self.X_placeholder, self.discriminator_parameters = _initialize_model(self.discriminator_layers, self.n_classes)
        self.Z_placeholder, self.generator_parameters = _initialize_model(self.generator_layers, self.n_classes)
        
        if y is None:
            generator_logit = _output_logit_tensor(self.Z_placeholder, self.generator_layers, self.generator_parameters)
            discriminator_logit_real = _output_logit_tensor(self.X_placeholder, self.discriminator_layers, self.discriminator_parameters)
            discriminator_logit_generated = _output_logit_tensor(tf.nn.sigmoid(generator_logit), self.discriminator_layers, self.discriminator_parameters)
        else:
            generator_logit = _output_logit_tensor(tf.concat(axis=1, values=[self.Z_placeholder, self.y_placeholder]), self.generator_layers, self.generator_parameters)
            discriminator_logit_real = _output_logit_tensor(tf.concat(axis=1, values=[self.X_placeholder, self.y_placeholder]), self.discriminator_layers, self.discriminator_parameters)
            discriminator_logit_generated = _output_logit_tensor(tf.concat(axis=1, values=[tf.nn.sigmoid(generator_logit), self.y_placeholder]), self.discriminator_layers, self.discriminator_parameters)
        
        self.discriminator_loss = _return_loss(discriminator_logit_real, True) + _return_loss(discriminator_logit_generated, False)
        self.generator_loss = _return_loss(discriminator_logit_generated, True)

        self.discriminator_update_parameters = _update_model_parameters(self.discriminator_optimizer, self.discriminator_loss, self.discriminator_parameters)
        self.generator_update_parameters = _update_model_parameters(self.generator_optimizer, self.generator_loss, self.generator_parameters)

        self.n_batches = int(X.shape[0] / batch_size)
        self.n_Z_features = self.generator_layers[0][0] - self.n_classes

    def _mini_batch_training(n_batches, ):
        for batch_index in range(self.n_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            X_batch = X_epoch[start_index:end_index]
            if y is not None:
                y_batch = y_epoch[start_index:end_index]
            if y is None:
                _, discriminator_loss_value = session.run([self.discriminator_update_parameters, self.discriminator_loss], feed_dict={self.X_placeholder: X_batch, self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features)})
            else:
                _, discriminator_loss_value = session.run([self.discriminator_update_parameters, self.discriminator_loss], feed_dict={self.X_placeholder: X_batch, self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features), self.y_placeholder: y_batch})
    
    def _start_training(self, X, y, nb_epoch, batch_size, discriminator_steps, verbose, session):
        for epoch in range(nb_epoch):
            for _ in range(discriminator_steps):
                epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
                X_epoch = X[epoch_shuffled_indices]
                if y is not None:
                    y_epoch = y[epoch_shuffled_indices]
                for batch_index in range(self.n_batches):
                    start_index = batch_index * batch_size
                    end_index = start_index + batch_size
                    X_batch = X_epoch[start_index:end_index]
                    if y is not None:
                        y_batch = y_epoch[start_index:end_index]
                    if y is None:
                        _, discriminator_loss_value = session.run([self.discriminator_update_parameters, self.discriminator_loss], feed_dict={self.X_placeholder: X_batch, self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features)})
                    else:
                        _, discriminator_loss_value = session.run([self.discriminator_update_parameters, self.discriminator_loss], feed_dict={self.X_placeholder: X_batch, self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features), self.y_placeholder: y_batch})
            epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
            X_epoch = X[epoch_shuffled_indices]
            if y is not None:
                y_epoch = y[epoch_shuffled_indices]
            for batch_index in range(self.n_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                X_batch = X_epoch[start_index:end_index]
                if y is not None:
                    y_batch = y_epoch[start_index:end_index]
                if y is None:
                    _, generator_loss_value = session.run([self.generator_update_parameters, self.generator_loss], feed_dict={self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features)})
                else:
                    _, generator_loss_value = session.run([self.generator_update_parameters, self.generator_loss], feed_dict={self.Z_placeholder: _sample_Z(batch_size, self.n_Z_features), self.y_placeholder: y_batch})
            print('Epoch: {}, discriminator loss: {}, generator loss: {}'.format(epoch, discriminator_loss_value, generator_loss_value))


class GAN(BaseGAN):

    def train(self, X, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        super()._prepare_training(X, None, batch_size)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            super()._start_training(X, None, nb_epoch, batch_size, discriminator_steps, verbose, sess)

class CGAN(BaseGAN):

    def train(self, X, y, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        super()._prepare_training(X, y, batch_size)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            super()._start_training(X, y, nb_epoch, batch_size, discriminator_steps, verbose, sess)