import tensorflow as tf

class PowerSeriesModelWrapper(tf.keras.layers.Layer):
    def __init__(self, K, coefficients=None, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        if coefficients is None:
            coefficients = [0.0j] * K
        self.coefficients = tf.Variable(coefficients, dtype=tf.complex64)

    def build(self, input_shape):
        pass

    def call(self, x_time):
        x_time_complex = tf.cast(x_time, dtype=tf.complex64)
        y = tf.zeros_like(x_time_complex)

        for k in range(1, self.K + 1):
            abs_powered = tf.abs(x_time_complex) ** (k - 1)
            abs_powered_complex64 = tf.cast(abs_powered, dtype=tf.complex64)
            y += self.coefficients[k - 1] * x_time_complex * abs_powered_complex64

        return y