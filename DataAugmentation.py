import tensorflow as tf
import os
import cv2
import tools
import numpy as np
import sys


####### DATA AUGMENTATION ########
class DataAugOpts:
    apply_data_augmentation = False  # If false, none of the following options have any effect.
    horizontal_flip = False
    vertical_flip = False
    random_brightness = False
    brightness_prob = 0.5
    brightness_delta_lower = -32
    brightness_delta_upper = 32
    random_contrast = False
    contrast_prob = 0.5
    contrast_factor_lower = 0.5
    contrast_factor_upper = 1.5
    random_saturation = False
    saturation_prob = 0.5
    saturation_factor_lower = 0.5
    saturation_factor_upper = 1.5
    random_hue = False
    hue_prob = 0.5
    hue_delta_lower = -0.1
    hue_delta_upper = 0.1
    convert_to_grayscale_prob = 0
    write_image_after_data_augmentation = False
##################################


class DataAugmentation:
    def __init__(self, opts, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height
        self.data_aug_opts = args.data_aug_opts
        self.outdir = args.outdir
        self.write_image_after_data_augmentation = self.data_aug_opts.write_image_after_data_augmentation
        if args.num_workers > 1 and self.data_aug_opts.write_image_after_data_augmentation:
            raise Exception('Option write_image_after_data_augmentation is not compatible with more than one worker to load data.')


    def data_augmenter(self, image, bboxes, filename):
        # Photometric distortions:
        if self.data_aug_opts.random_brightness:
            image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower,
                                             self.data_aug_opts.brightness_delta_upper,
                                             self.data_aug_opts.brightness_prob)
        if self.data_aug_opts.random_contrast:
            image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower,
                                           self.data_aug_opts.contrast_factor_upper,
                                           self.data_aug_opts.contrast_prob)
        if self.data_aug_opts.random_saturation:
            image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower,
                                             self.data_aug_opts.saturation_factor_upper,
                                             self.data_aug_opts.saturation_prob)
        if self.data_aug_opts.random_hue:
            image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower,
                                      self.data_aug_opts.hue_delta_upper,
                                      self.data_aug_opts.hue_prob)
        if self.data_aug_opts.convert_to_grayscale_prob > 0:
            image = convert_to_grayscale(image, self.data_aug_opts.convert_to_grayscale_prob)
        # Flips:
        if self.data_aug_opts.horizontal_flip:
            flag = tf.random_uniform(()) < 0.5
            bboxes = tf.cond(flag, lambda: self.flip_boxes_horizontally(bboxes), lambda: tf.identity(bboxes))
            image = tf.cond(flag, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
        if self.data_aug_opts.vertical_flip:
            flag = tf.random_uniform(()) < 0.5
            bboxes = tf.cond(flag, lambda: self.flip_boxes_vertically(bboxes), lambda: tf.identity(bboxes))
            image = tf.cond(flag, lambda: tf.image.flip_up_down(image), lambda: tf.identity(image))
        # Write images (for verification):
        if self.write_image_after_data_augmentation:
            image = tf.py_func(self.write_image, [image, bboxes, filename], tf.float32)
            image.set_shape((None, None, 3))
        return image, bboxes, filename

    def flip_boxes_vertically(self, bboxes):
        # bboxes: (nboxes, 5)
        new_y_min = 1.0 - bboxes[:, 2] - bboxes[:, 4]  # (nboxes)
        before = bboxes[:, :2]  # (nboxes, 2)
        after = bboxes[:, 3:]  # (nboxes, 2)
        bboxes = tf.concat([before, tf.expand_dims(new_y_min, axis=1), after], axis=1)  # (nboxes, 5)
        return bboxes

    def flip_boxes_horizontally(self, bboxes):
        # bboxes: (nboxes, 5)
        new_x_min = 1.0 - bboxes[:, 1] - bboxes[:, 3]  # (nboxes)
        before = tf.expand_dims(bboxes[:, 0], axis=1)  # (nboxes, 2)
        after = bboxes[:, 2:]  # (nboxes, 2)
        bboxes = tf.concat([before, tf.expand_dims(new_x_min, axis=1), after], axis=1)  # (nboxes)
        return bboxes

    def write_image(self, image, bboxes, filename):
        filename_str = filename.decode(sys.getdefaultencoding())
        file_path_candidate = os.path.join(self.outdir, 'image_after_data_aug_' + filename_str + '.png')
        file_path = tools.ensure_new_path(file_path_candidate)
        print('path to save image: ' + file_path)
        print(str(np.min(image)) + '   ' + str(np.mean(image)) + '   ' + str(np.max(image)))
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tools.add_bounding_boxes_to_image(img, bboxes)
        cv2.imwrite(file_path, img)
        return image


def adjust_contrast(image, factor):
    image = tf.clip_by_value(127.5 + factor * (image - 127.5), 0, 255)
    return image


def random_adjust_contrast(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_contrast(image, factor), lambda: image)
    return image


def adjust_brightness(image, brightness_delta):
    image = tf.clip_by_value(tf.image.adjust_brightness(image, brightness_delta), 0, 255)
    return image


def random_adjust_brightness(image, delta_lower, delta_upper, prob):
    delta_brightness = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_brightness(image, delta_brightness), lambda: image)
    return image


def random_adjust_saturation(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_saturation(image, factor), lambda: image)
    return image


def random_adjust_hue(image, delta_lower, delta_upper, prob):
    delta_hue = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_hue(image, delta_hue), lambda: image)
    return image


def convert_to_grayscale(image, prob):
    image_gray = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: image_gray, lambda: image)
    return image


