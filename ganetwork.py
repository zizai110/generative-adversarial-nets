import numpy as np
import tensorflow as tf


OPTIMIZER = tf.train.AdamOptimizer()

def bind_columns(tensor1, tensor2):
    if tensor2 is None:
        return tensor1
    return tf.concat(axis=1, values=[tensor1, tensor2])

def initialize_model(model_layers, input_layer_correction):
    model_parameters = {}
    for layer_index in range(len(model_layers) - 1):
        model_parameters['W' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index][0], model_layers[layer_index + 1][0]]))
        model_parameters['b' + str(layer_index)] = tf.Variable(tf.random_uniform([model_layers[layer_index + 1][0]]))
    input_data_placeholder = tf.placeholder(tf.float32, shape=[None, model_layers[0][0] - input_layer_correction])
    return input_data_placeholder, model_parameters
    
def output_logits_tensor(input_tensor, model_layers, model_parameters):
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
    return np.random.uniform(-1., 1., size=[n_samples, n_features]).astype(np.float32)

def sample_y(n_samples, n_classes, class_label):
    y = np.zeros(shape=[n_samples, n_classes]).astype(np.float32)
    y[:, class_label] = 1.
    return y
    
def return_loss(logits, positive_class_labels=True):
    if positive_class_labels:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))
    else:
        loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))
    return loss

def update_model_parameters(optimizer, loss, model_parameters):
    return optimizer.minimize(loss, var_list=list(model_parameters.values()))

def shuffle_data(X, y):
    epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
    X_epoch = X[epoch_shuffled_indices]
    y_epoch = y[epoch_shuffled_indices] if y is not None else None
    return X_epoch, y_epoch

def create_mini_batch_data(X, y, mini_batch_indices):
    X_batch = X[slice(*mini_batch_indices)]
    y_batch = y[slice(*mini_batch_indices)] if y is not None else None
    return X_batch, y_batch
                        
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

    def _initialize_training_parameters(self, X, y, batch_size):
        self.n_classes = y.shape[1] if y is not None else 0
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.n_classes]) if y is not None else None
        self.X_placeholder, self.discriminator_parameters = initialize_model(self.discriminator_layers, self.n_classes)
        self.Z_placeholder, self.generator_parameters = initialize_model(self.generator_layers, self.n_classes)
        
        generator_logit = output_logits_tensor(bind_columns(self.Z_placeholder, self.y_placeholder), self.generator_layers, self.generator_parameters)
        discriminator_logit_real = output_logits_tensor(bind_columns(self.X_placeholder, self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        discriminator_logit_generated = output_logits_tensor(bind_columns(tf.nn.sigmoid(generator_logit), self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        
        self.discriminator_loss = return_loss(discriminator_logit_real, True) + return_loss(discriminator_logit_generated, False)
        self.generator_loss = return_loss(discriminator_logit_generated, True)

        self.discriminator_update_parameters = update_model_parameters(self.discriminator_optimizer, self.discriminator_loss, self.discriminator_parameters)
        self.generator_update_parameters = update_model_parameters(self.generator_optimizer, self.generator_loss, self.generator_parameters)

        self.n_X_samples = X.shape[0]
        self.n_batches = self.n_X_samples // batch_size
        self.n_Z_features = self.generator_layers[0][0] - self.n_classes

        self.discriminator_placeholders = [placeholder for placeholder in [self.X_placeholder, self.Z_placeholder, self.y_placeholder] if placeholder is not None]
        self.generator_placeholders = [placeholder for placeholder in [self.Z_placeholder, self.y_placeholder] if placeholder is not None]

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def _return_epoch_loss_value(self, X, y, batch_size, session, model_update_parameters, model_loss, placeholders):
        X_epoch, y_epoch = shuffle_data(X, y)
        mini_batch_indices = mini_batch_indices_generator(self.n_X_samples, batch_size)
        for batch_index in range(self.n_batches):
            X_batch, y_batch = create_mini_batch_data(X, y, next(mini_batch_indices))
            feed_dict = {self.X_placeholder: X_batch, self.Z_placeholder: sample_Z(batch_size, self.n_Z_features), self.y_placeholder: y_batch}
            feed_dict = {placeholder: data for placeholder, data in feed_dict.items() if placeholder in placeholders}
            _, loss_value = session.run([model_update_parameters, model_loss], feed_dict=feed_dict)
        return loss_value

    def _train_gan(self, X, y, nb_epoch, batch_size, discriminator_steps, verbose, session):
        for epoch in range(nb_epoch):
            for _ in range(discriminator_steps):
                discriminator_loss_value = self._return_epoch_loss_value(X, y, batch_size, session, self.discriminator_update_parameters, self.discriminator_loss, self.discriminator_placeholders)
            generator_loss_value = self._return_epoch_loss_value(X, y, batch_size, session, self.generator_update_parameters, self.generator_loss, self.generator_placeholders)

            print('Epoch: {}, discriminator loss: {}, generator loss: {}'.format(epoch, discriminator_loss_value, generator_loss_value))


class GAN(BaseGAN):

    def train(self, X, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        super()._initialize_training_parameters(X, None, batch_size)
        super()._train_gan(X, None, nb_epoch, batch_size, discriminator_steps, verbose, self.sess)
        return self

    def generate_samples(self, n_samples):
        input_tensor = sample_Z(n_samples, self.n_Z_features)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples

class CGAN(BaseGAN):

    def train(self, X, y, nb_epoch, batch_size, discriminator_steps=1, verbose=1):
        super()._initialize_training_parameters(X, y, batch_size)
        super()._train_gan(X, y, nb_epoch, batch_size, discriminator_steps, verbose, self.sess)
        return self

    def generate_samples(self, n_samples, class_label):
        input_tensor = np.concatenate([sample_Z(n_samples, self.n_Z_features), sample_y(n_samples, self.n_classes, class_label)], axis=1)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples
