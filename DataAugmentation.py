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
    hue_delta_lower = -30
    hue_delta_upper = 30
    convert_to_grayscale_prob = 0
    ssd_original_pipeline = False
    max_expand_scale = 3
    min_crop_scale = 0.1
    write_image_after_data_augmentation = False
##################################


class DataAugmentation:
    def __init__(self, opts, outdir):
        self.opts = opts
        self.outdir = outdir
        self.write_image_after_data_augmentation = self.opts.write_image_after_data_augmentation

    def data_augmenter(self, image, bboxes, filename):
        if self.opts.ssd_original_pipeline:
            image, bboxes = self.ssd_original_pipeline(image, bboxes)
        else:
            image, bboxes = self.custom_data_aug(image, bboxes)
        # Write images (for verification):
        if self.write_image_after_data_augmentation:
            self.write_image(image, bboxes, filename)
        assert np.max(image) <= 255, 'np.max(image) > 255'
        assert np.min(image) >= 0, 'np.min(image) < 0'
        return image, bboxes

    def custom_data_aug(self, image, bboxes):
        # Photometric distortions:
        image = image.astype(np.float32)
        if self.opts.random_brightness:
            image = random_adjust_brightness(image, self.opts.brightness_delta_lower, self.opts.brightness_delta_upper, self.opts.brightness_prob)
        if self.opts.random_contrast:
            image = random_adjust_contrast(image, self.opts.contrast_factor_lower, self.opts.contrast_factor_upper, self.opts.contrast_prob)
        if self.opts.random_saturation:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = random_adjust_saturation(image, self.opts.saturation_factor_lower, self.opts.saturation_factor_upper, self.opts.saturation_prob)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if self.opts.random_hue:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = random_adjust_hue(image, self.opts.hue_delta_lower, self.opts.hue_delta_upper, self.opts.hue_prob)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if self.opts.convert_to_grayscale_prob > 0:
            image = convert_to_grayscale(image, self.opts.convert_to_grayscale_prob)
        # Flips:
        if self.opts.horizontal_flip:
            image, bboxes = horizontal_flip(image, bboxes)
        if self.opts.vertical_flip:
            image, bboxes = vertical_flip(image, bboxes)
        return image, bboxes

    def photometrics_sequence1(self, image):
        # image is in the range [0, 255], in RGB (or BGR).
        image = image.astype(np.float32)
        image = random_adjust_brightness(image, self.opts.brightness_delta_lower, self.opts.brightness_delta_upper, self.opts.brightness_prob)
        image = random_adjust_contrast(image, self.opts.contrast_factor_lower, self.opts.contrast_factor_upper, self.opts.contrast_prob)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = random_adjust_saturation(image, self.opts.saturation_factor_lower, self.opts.saturation_factor_upper, self.opts.saturation_prob)
        image = random_adjust_hue(image, self.opts.hue_delta_lower, self.opts.hue_delta_upper, self.opts.hue_prob)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def photometrics_sequence2(self, image):
        image = image.astype(np.float32)
        image = random_adjust_brightness(image, self.opts.brightness_delta_lower, self.opts.brightness_delta_upper, self.opts.brightness_prob)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = random_adjust_saturation(image, self.opts.saturation_factor_lower, self.opts.saturation_factor_upper, self.opts.saturation_prob)
        image = random_adjust_hue(image, self.opts.hue_delta_lower, self.opts.hue_delta_upper, self.opts.hue_prob)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = random_adjust_contrast(image, self.opts.contrast_factor_lower, self.opts.contrast_factor_upper, self.opts.contrast_prob)
        return image

    def ssd_photometric_distortions(self, image):
        if np.random.rand() < 0.5:
            image = self.photometrics_sequence1(image)
        else:
            image = self.photometrics_sequence2(image)
        return image

    def ssd_original_pipeline(self, image, bboxes):
        # bboxes (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        # Photometric distortions:
        image = self.ssd_photometric_distortions(image)
        # Expand and crop:
        image, bboxes = expand_and_crop_ssd(image, bboxes, self.opts.max_expand_scale, self.opts.min_crop_scale)
        # Random flip:
        image, bboxes = horizontal_flip(image, bboxes)
        return image, bboxes

    def write_image(self, image, bboxes, filename):
        file_path_candidate = os.path.join(self.outdir, 'image_after_data_aug_' + filename + '.png')
        file_path = tools.ensure_new_path(file_path_candidate)
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tools.add_bounding_boxes_to_image(img, bboxes)
        cv2.imwrite(file_path, img)
        return


def horizontal_flip(image, bboxes):
    # bboxes (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        bboxes[:, 1] = 1.0 - bboxes[:, 1] - bboxes[:, 3]
    return image, bboxes


def vertical_flip(image, bboxes):
    # bboxes (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)
        bboxes[:, 2] = 1.0 - bboxes[:, 2] - bboxes[:, 4]
    return image, bboxes


def expand_and_crop_ssd(image, bboxes, max_expand_scale, min_crop_scale):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 7), [class_id, x_min, y_min, width, height, pc, gt_idx]

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
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_width).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_height).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_width).astype(np.int32), img_width - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_height).astype(np.int32), img_height - patch_y0_abs)
    # Image:
    patch = image[patch_y0_abs:(patch_y0_abs+patch_height_abs), patch_x0_abs:(patch_x0_abs+patch_width_abs), :]  # (patch_height_abs, patch_width_abs, 3)
    # Percent contained:
    pc = compute_pc(bboxes, [patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel])  # (n_gt)

    # Remaining boxes:
    remaining_boxes_mask = pc > 1e-4  # (n_gt)
    n_remaining_boxes = np.sum(remaining_boxes_mask.astype(np.int8))
    remaining_boxes_mask_ext = np.tile(np.expand_dims(remaining_boxes_mask, axis=1), [1, 7])  # (n_gt, 7)
    remaining_boxes_on_orig_coords = np.extract(remaining_boxes_mask_ext, bboxes)  # (n_remaining_boxes * 6)
    remaining_boxes_on_orig_coords = np.reshape(remaining_boxes_on_orig_coords, (n_remaining_boxes, 7))  # (n_remaining_boxes, 7)

    remaining_boxes_class_id = remaining_boxes_on_orig_coords[:, 0]  # (n_remaining_boxes)
    remaining_boxes_pc = np.extract(remaining_boxes_mask, pc)  # (n_remaining_boxes)
    orig_boxes_gt_idx = bboxes[:, 6]  # (n_gt)
    remaining_boxes_gt_idx = np.extract(remaining_boxes_mask, orig_boxes_gt_idx)  # (n_remaining_boxes)

    remaining_boxes_x0 = (remaining_boxes_on_orig_coords[:, 1] - patch_x0_rel) / patch_width_rel
    remaining_boxes_y0 = (remaining_boxes_on_orig_coords[:, 2] - patch_y0_rel) / patch_height_rel
    remaining_boxes_x1 = (remaining_boxes_on_orig_coords[:, 1] + remaining_boxes_on_orig_coords[:, 3] - patch_x0_rel) / patch_width_rel
    remaining_boxes_y1 = (remaining_boxes_on_orig_coords[:, 2] + remaining_boxes_on_orig_coords[:, 4] - patch_y0_rel) / patch_height_rel
    remaining_boxes_x0 = np.maximum(remaining_boxes_x0, 0)  # (n_remaining_boxes)
    remaining_boxes_y0 = np.maximum(remaining_boxes_y0, 0)  # (n_remaining_boxes)
    remaining_boxes_x1 = np.minimum(remaining_boxes_x1, 1)
    remaining_boxes_y1 = np.minimum(remaining_boxes_y1, 1)
    remaining_boxes_width = remaining_boxes_x1 - remaining_boxes_x0  # (n_remaining_boxes)
    remaining_boxes_height = remaining_boxes_y1 - remaining_boxes_y0  # (n_remaining_boxes)

    remaining_boxes = np.stack([remaining_boxes_class_id, remaining_boxes_x0,
                                remaining_boxes_y0, remaining_boxes_width,
                                remaining_boxes_height, remaining_boxes_pc,
                                remaining_boxes_gt_idx], axis=-1)  # (n_remaining_boxes, 7)
    return patch, remaining_boxes


def compute_pc(gt_boxes, patch):
    # patch: [xmin, ymin, width, height]
    # gt_boxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]

    n_gt = gt_boxes.shape[0]
    patch_exp = np.expand_dims(patch, axis=0)
    patch_exp = np.tile(patch_exp, [n_gt, 1])  # (n_gt, 4)

    patch_xmin = patch_exp[:, 0]  # (n_gt)
    patch_ymin = patch_exp[:, 1]  # (n_gt)
    patch_xmax = patch_exp[:, 0] + patch_exp[:, 2]  # (n_gt)
    patch_ymax = patch_exp[:, 1] + patch_exp[:, 3]  # (n_gt)

    gt_boxes_xmin = gt_boxes[:, 1]  # (n_gt)
    gt_boxes_ymin = gt_boxes[:, 2]  # (n_gt)
    gt_boxes_xmax = gt_boxes[:, 1] + gt_boxes[:, 3]  # (n_gt)
    gt_boxes_ymax = gt_boxes[:, 2] + gt_boxes[:, 4]  # (n_gt)

    intersec_xmin = np.maximum(patch_xmin, gt_boxes_xmin)  # (n_gt)
    intersec_ymin = np.maximum(patch_ymin, gt_boxes_ymin)  # (n_gt)
    intersec_xmax = np.minimum(patch_xmax, gt_boxes_xmax)  # (n_gt)
    intersec_ymax = np.minimum(patch_ymax, gt_boxes_ymax)  # (n_gt)

    zero_grid = np.zeros((n_gt))  # (n_gt)
    w = np.maximum(intersec_xmax - intersec_xmin, zero_grid)  # (n_gt)
    h = np.maximum(intersec_ymax - intersec_ymin, zero_grid)  # (n_gt)
    area_intersec = w * h  # (n_gt)

    area_gt = gt_boxes[:, 3] * gt_boxes[:, 4]  # (n_gt)
    pc = area_intersec / area_gt  # (n_gt)

    pc = pc * np.array(gt_boxes[:, 5], dtype=np.float32)

    return pc


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
    image[:, :, 1] = np.clip(image[:, :, 1] * factor, 0, 1)
    return image


def random_adjust_saturation(image, factor_lower, factor_upper, prob):
    # image is in HSV.
    factor = np.random.rand() * (factor_upper - factor_lower) + factor_lower
    if np.random.rand() < prob:
        image = adjust_saturation(image, factor)
    return image


def adjust_hue(image, delta):
    # image is in HSV.
    image[:, :, 0] = np.clip(image[:, :, 0] + delta, 0, 359)
    return image


def random_adjust_hue(image, delta_lower, delta_upper, prob):
    # image is in HSV.
    delta = np.random.rand() * (delta_upper - delta_lower) + delta_lower
    if np.random.rand() < prob:
        image = adjust_hue(image, delta)
    return image


def convert_to_grayscale(image, prob):
    if np.random.rand() < prob:
        image = np.tile(np.sum(image, axis=-1, keepdims=True) / 3.0, [1, 1, 3])
    return image


