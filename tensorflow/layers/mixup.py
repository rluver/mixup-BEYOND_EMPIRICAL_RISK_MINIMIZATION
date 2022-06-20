import tensorflow as tf
from tensorflow.keras.layers import Layer


class MixUp(Layer):
    def __init__(self, alpha1=0.2, alpha2=0.2):
        super(MixUp, self).__init__()
        self.alpha1=alpha1
        self.alpha2=alpha2

    def _sample_beta_distribution(self):
        gamma_1_sample = tf.random.gamma(shape=(self.batch_size,), alpha=self.alpha1)
        gamma_2_sample = tf.random.gamma(shape=(self.batch_size,), alpha=self.alpha2)

        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def call(self, data1, data2):
        images_one, labels_one = data1
        images_two, labels_two = data2
        self.batch_size = tf.shape(images_one)[0]

        l = self._sample_beta_distribution()
        x_l = tf.reshape(l, (self.batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (self.batch_size, 1))
        
        new_images = images_one * x_l + images_two * (1 - x_l)
        new_labels = labels_one * y_l + labels_two * (1 - y_l)
        
        return new_images, new_labels
