import cv2
import numpy as np
import network


class ImageCropperOptions:
    def __init__(self):
        self.padding_factor = 0.25
        self.min_side_scale = 0.05
        self.max_side_scale = 0.7


class ImageCropper:
    def __init__(self, image_cropper_opts, single_cell_arch, n_crops_per_image):
        self.opts = image_cropper_opts
        self.single_cell_arch = single_cell_arch
        self.n_crops_per_image = n_crops_per_image

    def take_crops_on_image(self, image, bboxes):
        # Pad the image to make it square:
        image, bboxes = pad_to_make_square(image, bboxes)
        # Add a black frame around the image, to allow crops to lie partially outside the image:
        padded_image, bboxes, padded_side = self.pad_image(image, bboxes)
        # Make crops:
        crops = np.zeros(shape=(self.n_crops_per_image, network.receptive_field_size, network.receptive_field_size, 3), dtype=np.float32)
        labels_enc = np.zeros(shape=(self.n_crops_per_image, self.single_cell_arch.n_labels), dtype=np.float32)
        for i in range(self.n_crops_per_image):
            this_crop, this_label_enc = self.make_crop_keep_one_box_and_resize(padded_image, bboxes, padded_side)
            crops[i, :, :, :] = this_crop
            labels_enc[i, :] = this_label_enc
        return crops, labels_enc

    def pad_image(self, image, bboxes):
        # image: (sq_side, sq_side, 3)
        sq_side, width, _ = image.shape
        assert sq_side == width, 'Image is not square in pad_image.'
        padded_side = sq_side * (1 + 2 * self.opts.padding_factor)
        increment_top = int(np.round((padded_side - sq_side) / 2))
        increment_bottom = int(padded_side - sq_side - increment_top)
        increment_left = int(np.round((padded_side - sq_side) / 2))
        increment_right = int(padded_side - sq_side - increment_left)
        padded_image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
        # Warp and shift boxes:
        rel_incr_left = float(increment_left) / sq_side
        rel_incr_right = float(increment_right) / sq_side
        rel_incr_top = float(increment_top) / sq_side
        rel_incr_bottom = float(increment_bottom) / sq_side
        bboxes[:, 1] = (bboxes[:, 1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
        bboxes[:, 2] = (bboxes[:, 2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
        bboxes[:, 3] = bboxes[:, 3] / (1.0 + rel_incr_left + rel_incr_right)
        bboxes[:, 4] = bboxes[:, 4] / (1.0 + rel_incr_top + rel_incr_bottom)
        return padded_image, bboxes, padded_side

    def make_crop_keep_one_box_and_resize(self, image, bboxes, img_side):
        # image: (img_side, img_side, 3)
        # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
        # Take a random crop:
        patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale, self.opts.max_side_scale)
        # Encode the labels (and keep only one box):
        label_enc = self.single_cell_arch.encode_gt_from_array(remaining_boxes)  # (9)
        # Resize the crop to the size expected by the network:
        crop = cv2.resize(patch, (network.receptive_field_size, network.receptive_field_size), interpolation=1)  # (receptive_field_size, receptive_field_size, 3)
        return crop, label_enc


def pad_to_make_square(image, bboxes):
    # image: (orig_height, orig_width, 3)
    # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
    orig_height, orig_width, _ = image.shape
    max_side = max(orig_height, orig_width)
    # Increment on each side:
    increment_height = int(max_side - orig_height)
    increment_top = int(np.round(increment_height / 2.0))
    increment_bottom = increment_height - increment_top
    increment_width = int(max_side - orig_width)
    increment_left = int(np.round(increment_width / 2.0))
    increment_right = increment_width - increment_left
    image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
    # Warp and shift boxes:
    rel_incr_left = float(increment_left) / orig_width
    rel_incr_right = float(increment_right) / orig_width
    rel_incr_top = float(increment_top) / orig_height
    rel_incr_bottom = float(increment_bottom) / orig_height
    bboxes[:, 1] = (bboxes[:, 1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
    bboxes[:, 2] = (bboxes[:, 2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
    bboxes[:, 3] = bboxes[:, 3] / (1.0 + rel_incr_left + rel_incr_right)
    bboxes[:, 4] = bboxes[:, 4] / (1.0 + rel_incr_top + rel_incr_bottom)
    return image, bboxes


def sample_random_patch(image, bboxes, img_side, min_scale, max_scale):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
    patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(img_side, min_scale, max_scale)
    patch, remaining_boxes = sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 6) [class_id, xmin, ymin, width, height, percent_contained]
    return patch, remaining_boxes


def sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 6) [class_id, xmin, ymin, width, height, percent_contained]
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_side).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_side).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_side).astype(np.int32), img_side - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_side).astype(np.int32), img_side - patch_y0_abs)
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


def make_patch_shape(img_side, min_scale, max_scale):
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    patch_side = scale * img_side
    patch_side = np.minimum(np.maximum(np.round(patch_side), 1), img_side).astype(np.int32)
    x0 = np.random.randint(img_side - patch_side + 1)
    y0 = np.random.randint(img_side - patch_side + 1)
    # Convert to relative coordinates:
    x0 = x0 / float(img_side)
    y0 = y0 / float(img_side)
    patch_width = patch_side / float(img_side)
    patch_height = patch_side / float(img_side)
    return x0, y0, patch_width, patch_height




