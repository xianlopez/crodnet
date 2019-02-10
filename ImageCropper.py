import cv2
import numpy as np
import network


class ImageCropperOptions:
    def __init__(self, n_crops_per_image):
        self.n_crops_per_image = n_crops_per_image
        self.padding_factor = 0.5
        self.min_side_scale = 0.2
        self.max_side_scale = 0.9


class ImageCropper:
    def __init__(self, image_cropper_opts, single_cell_arch):
        self.opts = image_cropper_opts
        self.single_cell_arch = single_cell_arch

    def take_crops_on_image(self, image, bboxes):
        orig_height, orig_width, _ = image.shape
        padded_image, padded_height, padded_width = self.pad_image(image, orig_height, orig_width)
        crops = np.zeros(shape=(self.opts.n_crops_per_image, network.receptive_field_size, network.receptive_field_size, 3), dtype=np.float32)
        labels_enc = np.zeros(shape=(self.opts.n_crops_per_image, self.single_cell_arch.n_labels), dtype=np.float32)
        for i in range(self.opts.n_crops_per_image):
            this_crop, this_label_enc = self.make_crop_keep_one_box_and_resize(padded_image, bboxes, padded_width, padded_height)
            crops[i, :, :, :] = this_crop
            labels_enc[i, :] = this_label_enc
        return crops, labels_enc

    def pad_image(self, image, orig_height, orig_width):
        padded_height = orig_height * (1 + 2 * self.opts.padding_factor)
        padded_width = orig_width * (1 + 2 * self.opts.padding_factor)
        increment_top = int(np.round((padded_height - orig_height) / 2))
        increment_bottom = padded_height - orig_height - increment_top
        increment_left = int(np.round((padded_width - orig_width) / 2))
        increment_right = padded_width - orig_width - increment_left
        padded_image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
        return padded_image, padded_height, padded_width

    def make_crop_keep_one_box_and_resize(self, image, bboxes, img_width, img_height):
        # image: (height, width, 3)
        # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
        # Take a random crop:
        patch, remaining_boxes = sample_random_patch(image, bboxes, img_width, img_height, self.opts.min_side_scale, self.opts.max_side_scale)
        # Encode the labels (and keep only one box):
        label_enc = self.single_cell_arch.encode_gt_from_array(bboxes)  # (9)
        # Resize the crop to the size expected by the network:
        crop = cv2.resize(patch, (network.receptive_field_size, network.receptive_field_size), interpolation=1)  # (receptive_field_size, receptive_field_size, 3)
        return crop, label_enc


def sample_random_patch(image, bboxes, img_width, img_height, min_scale, max_scale):
    # image: (height, width, 3)
    # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
    patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(img_width, img_height, min_scale, max_scale)
    patch, remaining_boxes = sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 6) [class_id, xmin, ymin, width, height, percent_contained]
    return patch, remaining_boxes


def sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    # image: (height, width, 3)
    # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_width).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_height).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_width).astype(np.int32), img_width - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_height).astype(np.int32), img_height - patch_y0_abs)
    assert patch_width_abs == patch_height_abs, 'Patch has different width and height.'
    patch_side_abs = patch_height_abs
    # Image:
    patch = image[patch_y0_abs:(patch_y0_abs+patch_side_abs), patch_x0_abs:(patch_x0_abs+patch_side_abs), :]  # (patch_side_abs, patch_side_abs, 3)
    # Percent contained:
    pc = compute_pc(bboxes, [patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel])  # (n_gt)
    # Bounding boxes:

    remaining_boxes_mask = pc > 1e-4  # (n_gt)
    n_remaining_boxes = np.sum(remaining_boxes_mask.astype(np.int8))
    remaining_boxes_mask_ext = np.tile(np.expand_dims(remaining_boxes_mask, axis=1), [1, 6])  # (n_gt, 6)
    remaining_boxes_orig_coords = np.extract(remaining_boxes_mask_ext, bboxes)  # (n_remaining_boxes * 6)
    remaining_boxes_orig_coords = np.reshape(remaining_boxes_orig_coords, (n_remaining_boxes, 6))  # (n_remaining_boxes, 6)

    remaining_boxes_class_id = remaining_boxes_orig_coords[:, 0]  # (n_remaining_boxes)
    remaining_boxes_pc = remaining_boxes_orig_coords[:, 5]  # (n_remaining_boxes)

    remaining_boxes_x0 = (remaining_boxes_orig_coords[:, 1] - patch_x0_rel) / patch_width_rel
    remaining_boxes_y0 = (remaining_boxes_orig_coords[:, 2] - patch_y0_rel) / patch_height_rel
    remaining_boxes_x1 = (remaining_boxes_orig_coords[:, 1] + remaining_boxes_orig_coords[:, 3] - patch_x0_rel) / patch_width_rel
    remaining_boxes_y1 = (remaining_boxes_orig_coords[:, 2] + remaining_boxes_orig_coords[:, 4] - patch_y0_rel) / patch_height_rel
    remaining_boxes_x0 = np.maximum(remaining_boxes_x0, 0)  # (n_remaining_boxes)
    remaining_boxes_y0 = np.maximum(remaining_boxes_y0, 0)  # (n_remaining_boxes)
    remaining_boxes_x1 = np.minimum(remaining_boxes_x1, 1)
    remaining_boxes_y1 = np.minimum(remaining_boxes_y1, 1)
    remaining_boxes_width = remaining_boxes_x1 - remaining_boxes_x0  # (n_remaining_boxes)
    remaining_boxes_height = remaining_boxes_y1 - remaining_boxes_y0  # (n_remaining_boxes)

    remaining_boxes = np.stack([remaining_boxes_class_id, remaining_boxes_x0,
                                remaining_boxes_y0, remaining_boxes_width,
                                remaining_boxes_height, remaining_boxes_pc], axis=-1)  # (n_remaining_boxes, 6)
    return patch, remaining_boxes


def compute_pc(gt_boxes, patch):
    # patch: [xmin, ymin, width, height]
    # gt_boxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]

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


def make_patch_shape(img_width, img_height, min_scale, max_scale):
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    patch_side = np.sqrt(scale * float(img_width) * float(img_height))
    patch_side = np.minimum(np.maximum(np.round(patch_side), 1), img_width).astype(np.int32)
    x0 = np.random.randint(img_width - patch_side + 1)
    y0 = np.random.randint(img_height - patch_side + 1)
    # Convert to relative coordinates:
    x0 = x0 / float(img_width)
    y0 = y0 / float(img_height)
    patch_width = patch_side / float(img_width)
    patch_height = patch_side / float(img_height)
    return x0, y0, patch_width, patch_height




