import numpy as np
import tensorflow as tf


OPTIMIZER = tf.train.AdamOptimizer()

class BaseGAN:

    def __init__(self, discriminator_layers, generator_layers, discriminator_optimizer=OPTIMIZER, generator_optimizer=OPTIMIZER):
        self.discriminator_layers = discriminator_layers
        self.generator_layers = generator_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    @staticmethod
    def _initialize_model(model_layers, input_layer_correction=0):
        model_parameters = {}
        for layer_index in range(len(model_layers) - 1):
            model_parameters['W' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index][0], model_layers[layer_index + 1][0]]))
            model_parameters['b' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index + 1][0]]))
        input_data_placeholder = tf.placeholder(tf.float32, shape=[None, model_layers[0][0] - input_layer_correction])
        return input_data_placeholder, model_parameters
    
    @staticmethod
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

    @staticmethod
    def _sample_Z(n_samples, n_features):
        return np.random.uniform(-1., 1., size=[n_samples, n_features]).astype(np.float32)

    @staticmethod
    def _return_loss(logits, positive_class_labels=True):
        if positive_class_labels:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))
        else:
            loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))
        return loss

    @staticmethod
    def _update_model_parameters(optimizer, loss, model_parameters):
        return optimizer.minimize(loss, var_list=list(model_parameters.values()))


class GAN(BaseGAN):

    def train(self, X, nb_epoch, batch_size, verbose=1):
        X_placeholder, discriminator_parameters = GAN._initialize_model(self.discriminator_layers)
        Z_placeholder, generator_parameters = GAN._initialize_model(self.generator_layers)
        
        generator_logit = GAN._output_logit_tensor(Z_placeholder, self.generator_layers, generator_parameters)
        discriminator_logit_real = GAN._output_logit_tensor(X_placeholder, self.discriminator_layers, discriminator_parameters)
        discriminator_logit_generated = GAN._output_logit_tensor(tf.nn.sigmoid(generator_logit), self.discriminator_layers, discriminator_parameters)

        discriminator_loss = GAN._return_loss(discriminator_logit_real, True) + GAN._return_loss(discriminator_logit_generated, False)
        generator_loss = GAN._return_loss(discriminator_logit_generated, True)

        discriminator_update_step = GAN._update_model_parameters(self.discriminator_optimizer, discriminator_loss, discriminator_parameters)
        generator_update_step = GAN._update_model_parameters(self.generator_optimizer, generator_loss, generator_parameters)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        n_batches = int(X.shape[0] / batch_size)
        n_Z_features = self.generator_layers[0][0]

        for epoch in range(nb_epoch):
            X_epoch = X[np.random.permutation(range(X.shape[0]))]
            for batch_index in range(n_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                X_batch = X_epoch[start_index:end_index]
                _, discriminator_loss_value = sess.run([discriminator_update_step, discriminator_loss], feed_dict={X_placeholder: X_batch, Z_placeholder: GAN._sample_Z(batch_size, n_Z_features)})
                _, generator_loss_value = sess.run([generator_update_step, generator_loss], feed_dict={Z_placeholder: GAN._sample_Z(batch_size, n_Z_features)})
            print('Epoch: {}, discriminator loss: {}, generator loss: {}'.format(epoch, discriminator_loss_value, generator_loss_value))
        
        sess.close()


class CGAN(BaseGAN):

    @staticmethod
    def _sample_y(n_samples, n_classes, class_label):
        y = np.zeros(shape=[n_samples, n_classes]).astype(np.float32)
        y[:, class_label] = 1.
        return y

    def train(self, X, y, nb_epoch, batch_size, verbose=1):
        n_classes = y.shape[1]
        X_placeholder, discriminator_parameters = CGAN._initialize_model(self.discriminator_layers, n_classes)
        Z_placeholder, generator_parameters = CGAN._initialize_model(self.generator_layers, n_classes)
        y_placeholder = tf.placeholder(tf.float32, [None, n_classes])
        
        generator_logit = CGAN._output_logit_tensor(tf.concat(axis=1, values=[Z_placeholder, y_placeholder]), self.generator_layers, generator_parameters)
        discriminator_logit_real = CGAN._output_logit_tensor(tf.concat(axis=1, values=[X_placeholder, y_placeholder]), self.discriminator_layers, discriminator_parameters)
        discriminator_logit_generated = CGAN._output_logit_tensor(tf.concat(axis=1, values=[tf.nn.sigmoid(generator_logit), y_placeholder]), self.discriminator_layers, discriminator_parameters)

        discriminator_loss = CGAN._return_loss(discriminator_logit_real, True) + GAN._return_loss(discriminator_logit_generated, False)
        generator_loss = CGAN._return_loss(discriminator_logit_generated, True)

        discriminator_update_step = CGAN._update_model_parameters(self.discriminator_optimizer, discriminator_loss, discriminator_parameters)
        generator_update_step = CGAN._update_model_parameters(self.generator_optimizer, generator_loss, generator_parameters)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        n_batches = int(X.shape[0] / batch_size)
        n_Z_features = self.generator_layers[0][0] - n_classes

        for epoch in range(nb_epoch):
            epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
            X_epoch, y_epoch = X[epoch_shuffled_indices], y[epoch_shuffled_indices]
            for batch_index in range(n_batches):
                start_index = batch_index * batch_size
                end_index = start_index + batch_size
                X_batch, y_batch = X_epoch[start_index:end_index], y_epoch[start_index:end_index]
                _, discriminator_loss_value = sess.run([discriminator_update_step, discriminator_loss], feed_dict={X_placeholder: X_batch, Z_placeholder: CGAN._sample_Z(batch_size, n_Z_features), y_placeholder: y_batch})
                _, generator_loss_value = sess.run([generator_update_step, generator_loss], feed_dict={Z_placeholder: CGAN._sample_Z(batch_size, n_Z_features), y_placeholder: y_batch})
            print('Epoch: {}, discriminator loss: {}, generator loss: {}'.format(epoch, discriminator_loss_value, generator_loss_value))

        sess.close()