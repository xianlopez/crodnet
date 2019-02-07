import tensorflow as tf

# VGG_MEAN = [123.68, 116.78, 103.94]
VGG_MEAN = [123.0, 117.0, 104.0]


class Preprocessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def subtract_mean(self, image):
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        image = image - means
        return image