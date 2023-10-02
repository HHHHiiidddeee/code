import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential


class Residual(tf.keras.Model):
    """
    Class which inherits from tensorflow.keras.Model class and forms residual block.
    """

    def __init__(self, num_channels, batch_norm=False, use_skip=False, mean_pool=False):
        """
        Constructor which creates an object of the class Residual.

        param num_channels: The number of channels in the convolution layer.
        param batch_norm: The boolean value for batch normalization.
        param use_skip: The boolean value for skip connection.
        param mean_pool: The boolean value for mean pooling.
        """

        super().__init__()
        self.batch_norm = batch_norm
        self.use_skip = use_skip
        self.mean_pool = mean_pool
        self.conv1 = layers.Conv2D(num_channels, (3, 3), strides=(1, 1), padding="same")
        self.conv2 = layers.Conv2D(num_channels, (3, 3), strides=(1, 1), padding="same",
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.0001))
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.pool = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")

    def call(self, X):
        """
        Method to feed input into the object of Residual.

        param X: Input value.
        return: Output value.
        """

        y = y_skip = self.conv1(X)
        if self.batch_norm:
            y = self.bn1(y)
        y = self.conv2(tf.keras.activations.relu(y))
        if self.batch_norm:
            y = self.bn2(y)
        if self.use_skip:
            y += y_skip
        y = tf.keras.activations.relu(y)
        if self.mean_pool:
            y = self.pool(y)
        return y


class ResnetBlock(tf.keras.layers.Layer):
    """
    Class which inherits from tensorflow.keras.layers.Layer class and forms resnet block.
    """

    def __init__(self, num_channels, num_blocks, batch_norm=False, use_skip=False, mean_pool=False):
        """
        Constructor which creates an object of the class ResnetBlock.

        param num_channels: The number of channels in the convolution layer in the first residual block.
        param num_blocksk: The number of redidual blocks.
        param batch_norm: The boolean value for batch normalization.
        param use_skip: The boolean value for skip connection.
        param mean_pool: The boolean value for mean pooling.
        """

        super().__init__()
        self.residual_layers = []
        for i in range(num_blocks):
            self.residual_layers.append(Residual(num_channels * (2 ** i), batch_norm, use_skip, mean_pool))

    def call(self, X):
        """
        Method to feed input into the object of ResnetBlock.

        param X: Input value.
        return: Output value.
        """

        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


def build_cnn_resnet(num_classes, num_channels=32, num_blocks=4, batch_norm=False, use_skip=False, mean_pool=False):
    model = Sequential()
    model.add(ResnetBlock(num_channels, num_blocks, batch_norm, use_skip, mean_pool))
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(216, activation="relu"))
    model.add(layers.Dropout(0.3, input_shape=(216,)))
    model.add(layers.Dense(units=num_classes, activation= 'softmax'))
    return model
