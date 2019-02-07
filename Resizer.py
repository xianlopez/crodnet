import tensorflow as tf
import cv2
import numpy as np


class ResizerWithLabels:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height

    def get_resize_func(self, resize_method):
        if resize_method == 'resize_warp':
            return self.resize_warp
        elif resize_method == 'resize_pad_zeros':
            return self.resize_pad_zeros
        else:
            raise Exception('Resize method not recognized.')

    def resize_warp(self, image, label):
        # The bounding boxes are in relative coordinates, so they don't need to be changed.
        # image = tf.image.resize_images(image, [self.input_height, self.input_width])
        image = tf.py_func(self.pyfunc_resize, [image, label], (tf.float32))
        image.set_shape((self.input_height, self.input_width, 3))
        return image, label

    def pyfunc_resize(self, image, label):
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

    def pad_with_zeros(self, image, size, bboxes):
        # Increment on each side:
        increment_height = int(self.input_height - size[0])
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(self.input_width - size[1])
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
        # Warp and shift boxes:
        rel_incr_left = float(increment_left) / size[1]
        rel_incr_right = float(increment_right) / size[1]
        rel_incr_top = float(increment_top) / size[0]
        rel_incr_bottom = float(increment_bottom) / size[0]
        for i in range(len(bboxes)):
            bboxes[i][1] = (bboxes[i][1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][2] = (bboxes[i][2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
            bboxes[i][3] = bboxes[i][3] / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][4] = bboxes[i][4] / (1.0 + rel_incr_top + rel_incr_bottom)
        return image, bboxes


def ResizeNumpy(image, method, input_width, input_height):
    if method == 'resize_warp':
        image = cv2.resize(image, (input_width, input_height))
    elif method == 'resize_pad_zeros':
        height, width, _ = image.shape
        # Resize so it fits totally in the input size:
        scale_width = input_width / np.float32(width)
        scale_height = input_height / np.float32(height)
        scale = min(scale_width, scale_height)
        new_width = int(np.round(width * scale))
        new_height = int(np.round(height * scale))
        image = cv2.resize(image, (new_width, new_height))
        # Pad with zeros the remaining areas:
        increment_height = int(input_height - new_height)
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(input_width - new_width)
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right,
                                   cv2.BORDER_CONSTANT)
    else:
        raise Exception('Resize method not recognized.')
    return image

