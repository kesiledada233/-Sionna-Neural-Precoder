from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
import tensorflow as tf

num_conv_channels = 128

class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z)  # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z


class Neural_Precoder(Layer):
    def __init__(self, return_effective_channel=False, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._return_effective_channel = return_effective_channel

        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                  kernel_size=[3, 3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=16,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        x = inputs
        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        ###
        ### Transformations to bring x in the desired shapes
        ###

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        # z : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx*2]
        z = tf.concat([tf.math.real(x_precoded), tf.math.imag(x_precoded)], axis=-1)

        # z : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx*4(num_user_ant)]
        # z = tf.pad(z, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 8]], mode='CONSTANT', constant_values=0)

        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        z_real_part, z_imag_part = tf.split(z, num_or_size_splits=2, axis=-1)
        z = tf.complex(z_real_part, z_imag_part)

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        z = tf.transpose(z, [0, 1, 4, 2, 3])

        return z