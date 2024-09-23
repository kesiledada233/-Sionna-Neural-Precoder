import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from tensorflow.signal import ifftshift

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel, subcarrier_frequencies,  cir_to_ofdm_channel, ApplyOFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
from PA_PSM import PowerSeriesModelWrapper
from sionna.signal.utils import fft, ifft
from sionna.ofdm import RemoveNulledSubcarriers


############################################
## Channel configuration
carrier_frequency = 3.5e9 # Hz
delay_spread = 100e-9 # s
cdl_model = "B" # CDL model to use
speed = 10.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -5.0
ebno_db_max = 10.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 30e3 # Hz
fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 4 # 16-QAM
coderate = 0.5 # Coderate for LDPC code

############################################
## Neural receiver configuration
num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver

stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter

resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = 4,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)

# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)

ut_antenna = Antenna(polarization="single",
                     polarization_type="V",
                     antenna_pattern="38.901",
                     carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

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
        self._output_conv = Conv2D(filters=num_bits_per_symbol,
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
        z = tf.concat([tf.math.real(x_precoded), tf.math.imag(x_precoded),], axis=-1)

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

K_example = 3
coefficients_example = [1 + 0j, -0.02 - 0.01j, -0.049 - 0.023j]
binary_source = BinarySource()
encoder = LDPC5GEncoder(k, n)
mapper = Mapper("qam", num_bits_per_symbol)
rg_mapper = ResourceGridMapper(resource_grid)
power_series_model = PowerSeriesModelWrapper(K_example,coefficients_example)
neural_precoder = Neural_Precoder()

cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_antenna, bs_array, "downlink", min_speed=speed)
channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)
frequencies = subcarrier_frequencies(resource_grid.fft_size, resource_grid.subcarrier_spacing)
channel_freq = ApplyOFDMChannel(add_awgn=True)

removed_null_subc = RemoveNulledSubcarriers(resource_grid)
ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
demapper = Demapper("app", "qam", num_bits_per_symbol)
decoder = LDPC5GDecoder(encoder, hard_out=True)

batch_size = 128
# ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
# if len(ebno_db.shape) == 0:
#             ebno_db = tf.fill([batch_size], ebno_db)
ebno_db =10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
b = binary_source([batch_size, 1, resource_grid.num_streams_per_tx, k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
x_rg = neural_precoder(x_rg)
x_time = ifft(x_rg)
x_time_pa = power_series_model(x_time)
x_rg = fft(x_time_pa)

no_ = expand_to_rank(no, tf.rank(x_rg))
y, h = channel([x_rg, no_])

# h_hat, err_var = ls_est ([y, no])
h_hat = removed_null_subc(h)
err_var = 0.0
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
llr = demapper([x_hat, no_eff_])
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))
