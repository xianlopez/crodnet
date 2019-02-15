import tensorflow as tf
import cv2
import numpy as np


class ResizerWithLabels:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height

    def resize_warp(self, image, label):
        # The bounding boxes are in relative coordinates, so they don't need to be changed.
        # image = tf.image.resize_images(image, [self.input_height, self.input_width])
        image = tf.py_func(self.pyfunc_resize, [image], (tf.float32))
        image.set_shape((self.input_height, self.input_width, 3))
        return image, label

    def pyfunc_resize(self, image):
        image = image.astype(np.uint8)
        image = cv2.resize(image, (self.input_height, self.input_width), interpolation=1)
        image = image.astype(np.float32)
        return image

    def resize_pad_zeros(self, image, bboxes):
        # Resize image so the biggest side fits exactly in the input size:
        # height, width = tf.shape(image)[0], tf.shape(image)[1]
        width, height = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.minimum(scale_height, scale_width)
        tf.cast(tf.round(scale * tf.to_float(height)), tf.int32)
        new_height = tf.minimum(tf.cast(tf.round(scale * tf.to_float(height)), tf.int32), self.input_height)
        new_width = tf.minimum(tf.cast(tf.round(scale * tf.to_float(width)), tf.int32), self.input_width)
        # size = tf.stack([new_height, new_width])
        size = tf.stack([new_width, new_height])
        image = tf.image.resize_images(image, size)
        # Pad the image with zeros and modify accordingly the bounding boxes:
        (image, bboxes) = tf.py_func(self.pad_with_zeros, (image, size, bboxes), (tf.float32, tf.float32))
        image.set_shape((self.input_height, self.input_width, 3))
        bboxes.set_shape((None, 6))
        return image, bboxes

