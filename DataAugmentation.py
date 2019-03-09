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
    hue_delta_lower = -10
    hue_delta_upper = 10
    convert_to_grayscale_prob = 0
    write_image_after_data_augmentation = False
##################################


class DataAugmentation:
    def __init__(self, opts):
        self.data_aug_opts = opts.data_aug_opts
        self.outdir = opts.outdir
        self.write_image_after_data_augmentation = self.data_aug_opts.write_image_after_data_augmentation
        if opts.num_workers > 1 and self.data_aug_opts.write_image_after_data_augmentation:
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

    def photometrics_sequence1(self, image):
        # image is in the range [0, 255], in RGB (or BGR).
        image = image.astype(np.int32)
        image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower, self.data_aug_opts.brightness_delta_upper, self.data_aug_opts.brightness_prob)
        image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower, self.data_aug_opts.contrast_factor_upper, self.data_aug_opts.contrast_prob)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower, self.data_aug_opts.saturation_factor_upper, self.data_aug_opts.saturation_prob)
        image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower, self.data_aug_opts.hue_delta_upper, self.data_aug_opts.hue_prob)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def photometrics_sequence2(self, image):
        image = image.astype(np.int32)
        image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower, self.data_aug_opts.brightness_delta_upper, self.data_aug_opts.brightness_prob)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower, self.data_aug_opts.saturation_factor_upper, self.data_aug_opts.saturation_prob)
        image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower, self.data_aug_opts.hue_delta_upper, self.data_aug_opts.hue_prob)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower, self.data_aug_opts.contrast_factor_upper, self.data_aug_opts.contrast_prob)
        return image

    def ssd_photometric_distortions(self, image):
        if np.random.rand() < 0.5:
            image = self.photometrics_sequence1(image)
        else:
            image = self.photometrics_sequence2(image)
        return image

    def ssd_original_pipeline(self, image, bboxes):
        # Photometric distortions:
        image = self.ssd_photometric_distortions(image)
        # Expand and crop:
        (image, bboxes) = tf.py_func(expand_and_crop_ssd, [image, bboxes], (tf.float32, tf.float32))
        image.set_shape((None, None, 3))
        bboxes.set_shape((None, 5))
        # Random flip:
        flag = tf.random_uniform(()) < 0.5
        bboxes = tf.cond(flag, lambda: self.flip_boxes_horizontally(bboxes), lambda: tf.identity(bboxes))
        image = tf.cond(flag, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
        return image, bboxes

    def flip_boxes_vertically(self, bboxes):
        # bboxes: (nboxes, 7)
        new_y_min = 1.0 - bboxes[:, 2] - bboxes[:, 4]  # (nboxes)
        before = bboxes[:, :2]  # (nboxes, 2)
        after = bboxes[:, 3:]  # (nboxes, 2)
        bboxes = tf.concat([before, tf.expand_dims(new_y_min, axis=1), after], axis=1)  # (nboxes, 7)
        return bboxes

    def flip_boxes_horizontally(self, bboxes):
        # bboxes: (nboxes, 7)
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


def expand_and_crop_ssd(image, bboxes, max_expand_scale, min_crop_scale):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]

    img_height, img_width, _ = image.shape

    ####################
    ### Expand:
    rnd1 = np.random.randint(2)
    if rnd1 == 0:
        scale = np.random.rand() * (max_expand_scale - 1) + 1
        new_width = np.round(img_width * scale).astype(np.int32)
        new_height = np.round(img_height * scale).astype(np.int32)
        max_image_value = np.max(image)
        canvas_R = np.random.rand() * max_image_value * np.ones(shape=(new_height, new_width), dtype=np.float32)
        canvas_G = np.random.rand() * max_image_value * np.ones(shape=(new_height, new_width), dtype=np.float32)
        canvas_B = np.random.rand() * max_image_value * np.ones(shape=(new_height, new_width), dtype=np.float32)
        canvas = np.stack([canvas_R, canvas_G, canvas_B], axis=-1)  # (new_height, new_width, 3)
        pos_i = np.random.randint(new_height - img_height + 1)
        pos_j = np.random.randint(new_width - img_width + 1)
        canvas[pos_i:(pos_i+img_height), pos_j:(pos_j+img_width), :] = image
        image = canvas
        bboxes[:, 1] = (pos_j + bboxes[:, 1] * img_width) / float(new_width)
        bboxes[:, 2] = (pos_i + bboxes[:, 2] * img_height) / float(new_height)
        bboxes[:, 3] = bboxes[:, 3] / scale
        bboxes[:, 4] = bboxes[:, 4] / scale
    else:
        new_width = img_width
        new_height = img_height

    ####################
    ### Random crop:
    min_scale = min_crop_scale
    max_scale = 1
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    iou_th_list = [-1, 0.1, 0.3, 0.5, 0.7, 0.9]
    while True: # Keep going until we either find a valid patch or return the original image.
        if np.random.rand() >= (1 - 0.857):
            iou_th = iou_th_list[np.random.randint(len(iou_th_list))]
            for _ in range(50):
                patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(new_width, new_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
                patch_x1_rel = patch_x0_rel + patch_width_rel
                patch_y1_rel = patch_y0_rel + patch_height_rel
                # Check boxes' IOU:
                patch_is_valid = False
                for i in range(bboxes.shape[0]):
                    x_center = bboxes[i, 1] + float(bboxes[i, 3]) / 2
                    y_center = bboxes[i, 2] + float(bboxes[i, 4]) / 2
                    if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
                        if tools.compute_iou([patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel], bboxes[i, 1:5]) > iou_th:
                            patch_is_valid = True
                            break
                if patch_is_valid:
                    image, bboxes = sample_patch(image, bboxes, new_width, new_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
                    return image, bboxes

        else:
            return image, bboxes


def make_patch_shape(img_width, img_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio):
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    aspect_ratio = np.random.rand() * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
    patch_width = np.sqrt(aspect_ratio * scale * float(img_width) * float(img_height))
    patch_height = np.sqrt(scale * float(img_width) * float(img_height) / aspect_ratio)
    patch_width = np.minimum(np.maximum(np.round(patch_width), 1), img_width).astype(np.int32)
    patch_height = np.minimum(np.maximum(np.round(patch_height), 1), img_height).astype(np.int32)
    x0 = np.random.randint(img_width - patch_width + 1)
    y0 = np.random.randint(img_height - patch_height + 1)
    # Convert to relative coordinates:
    x0 = x0 / float(img_width)
    y0 = y0 / float(img_height)
    patch_width = patch_width / float(img_width)
    patch_height = patch_height / float(img_height)
    return x0, y0, patch_width, patch_height


def sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_width).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_height).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_width).astype(np.int32), img_width - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_height).astype(np.int32), img_height - patch_y0_abs)
    # Image:
    patch = image[patch_y0_abs:(patch_y0_abs+patch_height_abs), patch_x0_abs:(patch_x0_abs+patch_width_abs), :]
    # Bounding boxes:
    patch_x1_rel = patch_x0_rel + patch_width_rel
    patch_y1_rel = patch_y0_rel + patch_height_rel
    remaining_boxes_list = []
    for i in range(bboxes.shape[0]):
        x_center = bboxes[i, 1] + float(bboxes[i, 3]) / 2
        y_center = bboxes[i, 2] + float(bboxes[i, 4]) / 2
        # print([patch_x0_rel, x_center, patch_x1_rel, patch_y0_rel, y_center, patch_y1_rel])
        if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
            # print('box valid')
            new_box_x0 = (bboxes[i, 1] - patch_x0_rel) / patch_width_rel
            new_box_y0 = (bboxes[i, 2] - patch_y0_rel) / patch_height_rel
            new_box_x1 = (bboxes[i, 1] + bboxes[i, 3] - patch_x0_rel) / patch_width_rel
            new_box_y1 = (bboxes[i, 2] + bboxes[i, 4] - patch_y0_rel) / patch_height_rel
            new_box_x0 = max(new_box_x0, 0)
            new_box_y0 = max(new_box_y0, 0)
            new_box_x1 = min(new_box_x1, 1)
            new_box_y1 = min(new_box_y1, 1)
            new_box_width = new_box_x1 - new_box_x0
            new_box_height = new_box_y1 - new_box_y0
            remaining_boxes_list.append([bboxes[i, 0], new_box_x0, new_box_y0, new_box_width, new_box_height])
        else:
            pass
            # print('box not valid')
    nboxes_remaining = len(remaining_boxes_list)
    remaining_boxes = np.zeros(shape=(nboxes_remaining, 5), dtype=np.float32)
    for i in range(nboxes_remaining):
        remaining_boxes[i, 0] = remaining_boxes_list[i][0]
        remaining_boxes[i, 1] = remaining_boxes_list[i][1]
        remaining_boxes[i, 2] = remaining_boxes_list[i][2]
        remaining_boxes[i, 3] = remaining_boxes_list[i][3]
        remaining_boxes[i, 4] = remaining_boxes_list[i][4]
    return patch, remaining_boxes


def adjust_contrast(image, factor):
    image = np.clip(127.5 + factor * (image - 127.5), 0, 255)
    return image


def random_adjust_contrast(image, factor_lower, factor_upper, prob):
    factor = np.random.rand() * (factor_upper - factor_lower) + factor_lower
    if np.random.rand() < prob:
        image = adjust_contrast(image, factor)
    return image


def adjust_brightness(image, delta):
    image = np.clip(image + delta, 0, 255)
    return image


def random_adjust_brightness(image, delta_lower, delta_upper, prob):
    delta = np.random.rand() * (delta_upper - delta_lower) + delta_lower
    if np.random.rand() < prob:
        image = adjust_brightness(image, delta)
    return image


def adjust_saturation(image, factor):
    # image is in HSV.
    image[:, :, 1] = np.clip(image[:, :, 1] * factor, 0, 255)
    return image


def random_adjust_saturation(image, factor_lower, factor_upper, prob):
    # image is in HSV.
    factor = np.random.rand() * (factor_upper - factor_lower) + factor_lower
    if np.random.rand() < prob:
        image = adjust_saturation(image, factor)
    return image


def adjust_hue(image, delta):
    # image is in HSV.
    image[:, :, 0] = np.clip(image[:, :, 0] + delta, 0, 179)
    return image


def random_adjust_hue(image, delta_lower, delta_upper, prob):
    # image is in HSV.
    delta = np.random.rand() * (delta_upper - delta_lower) + delta_lower
    if np.random.rand() < prob:
        image = adjust_hue(image, delta)
    return image


def convert_to_grayscale(image, prob):
    if np.random.rand() < prob:
        image = np.sum(image, axis=-1) / 3.0
    return image


