import cv2
import numpy as np
import network


class ImageCropperOptions:
    def __init__(self):
        self.padding_factor = 0.25
        self.min_side_scale = 0.05
        self.max_side_scale = 0.7
        self.max_dc = 0.2
        self.min_ar = 0.05
        self.max_ar = 0.5
        self.max_dc_pair = 0.3
        self.min_ar_pair = 0.2
        self.probability_focus = 0.4
        self.probability_pair = 0.3
        self.probability_inside = 0.2


class ImageCropper:
    def __init__(self, image_cropper_opts, single_cell_arch, n_crops_per_image):
        self.opts = image_cropper_opts
        self.single_cell_arch = single_cell_arch
        self.n_crops_per_image = n_crops_per_image
        assert self.opts.probability_focus + self.opts.probability_pair + self.opts.probability_inside <= 1, 'Bad probabilities'

    def take_crops_on_image(self, image, bboxes, hns):
        # image: (orig_height, orig_width, 3)
        # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
        # hns: (n_hns, 7) [class_id, xmin, ymin, width, height, 1, -1]
        # Pad the image to make it square:
        image, bboxes, hns = pad_to_make_square(image, bboxes, hns)
        # Add a black frame around the image, to allow crops to lie partially outside the image:
        padded_image, bboxes, hns, padded_side = self.pad_image(image, bboxes, hns)
        # Make crops:
        crops = np.zeros(shape=(self.n_crops_per_image, network.receptive_field_size, network.receptive_field_size, 3), dtype=np.float32)
        labels_enc = np.zeros(shape=(self.n_crops_per_image, self.single_cell_arch.n_labels), dtype=np.float32)
        for i in range(self.n_crops_per_image):
            this_crop, this_label_enc = self.crop_following_policy(padded_image, bboxes, hns, padded_side)
            crops[i, :, :, :] = this_crop
            labels_enc[i, :] = this_label_enc
        return crops, labels_enc

    def pad_image(self, image, bboxes, hns):
        # image: (sq_side, sq_side, 3)
        sq_side, width, _ = image.shape
        assert sq_side == width, 'Image is not square in pad_image.'
        padded_side = int(np.round(sq_side * (1 + 2 * self.opts.padding_factor)))
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
        # The same on hard negative boxes:
        hns[:, 1] = (hns[:, 1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
        hns[:, 2] = (hns[:, 2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
        hns[:, 3] = hns[:, 3] / (1.0 + rel_incr_left + rel_incr_right)
        hns[:, 4] = hns[:, 4] / (1.0 + rel_incr_top + rel_incr_bottom)
        return padded_image, bboxes, hns, padded_side

    def crop_only_random(self, image, bboxes, img_side):
        # image: (img_side, img_side, 3)
        # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
        patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale, self.opts.max_side_scale, self.opts.padding_factor)
        crop, label_enc = self.keep_one_box_and_resize(patch, remaining_boxes)
        return crop, label_enc

    def crop_following_policy(self, image, bboxes, hns, img_side):
        # image: (img_side, img_side, 3)
        # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
        # hns: (n_hns, 7) [class_id, xmin, ymin, width, height, 1, -1]
        n_gt = bboxes.shape[0]
        n_hns = hns.shape[0]
        rnd1 = np.random.rand()
        if rnd1 < self.opts.probability_focus:
            if n_gt + n_hns >= 1:
                box_idx = np.random.randint(0, n_gt + n_hns)
                if box_idx < n_gt:
                    focus_obj = bboxes[box_idx]
                else:
                    focus_obj = hns[box_idx - n_gt]
                patch, remaining_boxes = sample_patch_focusing_on_object(image, bboxes, img_side, focus_obj,
                                                                         self.opts.max_dc, self.opts.min_ar, self.opts.max_ar)
            else:
                patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale,
                                                             self.opts.max_side_scale, self.opts.padding_factor)
        elif rnd1 < self.opts.probability_focus + self.opts.probability_pair:
            if n_gt >= 2:
                box_idx1 = np.random.randint(0, n_gt)
                difference = np.random.randint(1, n_gt)
                box_idx2 = np.remainder(box_idx1 + difference, n_gt)
                assert box_idx1 != box_idx2, 'Using same box when sampling focusing on pair.'
                patch, remaining_boxes = sample_patch_focusing_on_pair(image, bboxes, img_side, box_idx1, box_idx2,
                                                                       self.opts.max_dc_pair, self.opts.min_ar_pair)
            else:
                patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale,
                                                             self.opts.max_side_scale, self.opts.padding_factor)
        elif rnd1 < self.opts.probability_focus + self.opts.probability_pair + self.opts.probability_inside:
            if n_gt >= 1:
                box_idx = np.random.randint(0, n_gt)
                patch, remaining_boxes = sample_patch_inside_object(image, bboxes, img_side, box_idx)
            else:
                patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale,
                                                             self.opts.max_side_scale, self.opts.padding_factor)
        else:
            patch, remaining_boxes = sample_random_patch(image, bboxes, img_side, self.opts.min_side_scale,
                                                         self.opts.max_side_scale, self.opts.padding_factor)
        crop, label_enc = self.keep_one_box_and_resize(patch, remaining_boxes)
        return crop, label_enc

    def keep_one_box_and_resize(self, patch, remaining_boxes):
        # patch: (patch_side_abs, patch_side_abs, 3)
        # remaining_boxes: (n_remaining_boxes, 9) [class_id, xmin, ymin, width, height, percent_contained, gt_idx, c_x_unclipped, c_y_unclipped]
        # Encode the labels (and keep only one box):
        label_enc = self.single_cell_arch.encode_gt_from_array(remaining_boxes)  # (n_labels)
        # Resize the crop to the size expected by the network:
        crop = cv2.resize(patch, (network.receptive_field_size, network.receptive_field_size), interpolation=1)  # (receptive_field_size, receptive_field_size, 3)
        return crop, label_enc


def pad_to_make_square(image, bboxes, hns):
    # image: (orig_height, orig_width, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    # hns: (n_hns, 7) [class_id, xmin, ymin, width, height, 1, -1]
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
    # The same on hard negative boxes:
    hns[:, 1] = (hns[:, 1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
    hns[:, 2] = (hns[:, 2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
    hns[:, 3] = hns[:, 3] / (1.0 + rel_incr_left + rel_incr_right)
    hns[:, 4] = hns[:, 4] / (1.0 + rel_incr_top + rel_incr_bottom)
    return image, bboxes, hns


def sample_patch_inside_object(image, bboxes, img_side, obj_idx):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    patch_x0_rel, patch_y0_rel, patch_side_rel = make_patch_shape_inside_object(bboxes[obj_idx, :])
    patch, remaining_boxes = sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_side_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 9) [class_id, xmin, ymin, width, height, percent_contained, gt_idx, c_x_unclipped, c_y_unclipped]
    return patch, remaining_boxes


def make_patch_shape_inside_object(bbox):
    # bbox: (6) [class_id, xmin, ymin, width, height, percent_contained]
    # Everything is in relative coordinates.
    patch_center_x = np.random.rand() * bbox[3] + bbox[1]
    patch_center_y = np.random.rand() * bbox[4] + bbox[2]
    patch_side = np.random.rand() * max(bbox[3], bbox[4])  # The patch side is, as maximum, the biggest side of the GT.
    # Patch final coordinates:
    patch_xmin = max(patch_center_x - patch_side / 2.0, 0)
    patch_ymin = max(patch_center_y - patch_side / 2.0, 0)
    if patch_xmin + patch_side > 1:
        patch_side = 1 - patch_xmin
    if patch_ymin + patch_side > 1:
        patch_side = 1 - patch_ymin
    return patch_xmin, patch_ymin, patch_side


def sample_patch_focusing_on_pair(image, bboxes, img_side, obj1_idx, obj2_idx, max_dc, min_ar):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    patch_x0_rel, patch_y0_rel, patch_side_rel = make_patch_shape_focusing_on_pair(bboxes[obj1_idx, :], bboxes[obj2_idx, :], max_dc, min_ar)
    patch, remaining_boxes = sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_side_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 9) [class_id, xmin, ymin, width, height, percent_contained, gt_idx, c_x_unclipped, c_y_unclipped]
    return patch, remaining_boxes


def make_patch_shape_focusing_on_pair(bbox1, bbox2, max_dc, min_ar):
    # bbox1: (6) [class_id, xmin, ymin, width, height, percent_contained]
    # bbox2: (6) [class_id, xmin, ymin, width, height, percent_contained]
    xmin = min(bbox1[1], bbox2[1])
    ymin = min(bbox1[2], bbox2[2])
    xmax = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    ymax = max(bbox1[2] + bbox1[4], bbox2[2] + bbox2[4])
    width = xmax - xmin
    height = ymax - ymin
    box_combined = [-1, xmin, ymin, width, height, -1]
    patch_xmin, patch_ymin, patch_side = make_patch_shape_focusing_on_object(box_combined, max_dc, min_ar)
    return patch_xmin, patch_ymin, patch_side


def sample_patch_focusing_on_object(image, bboxes, img_side, focus_obj, max_dc, min_ar, max_ar):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    patch_x0_rel, patch_y0_rel, patch_side_rel = make_patch_shape_focusing_on_object(focus_obj, max_dc, min_ar, max_ar)
    patch, remaining_boxes = sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_side_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 9) [class_id, xmin, ymin, width, height, percent_contained, gt_idx, c_x_unclipped, c_y_unclipped]
    return patch, remaining_boxes


def make_patch_shape_focusing_on_object(bbox, max_dc, min_ar, max_ar=1.0):
    # bbox: (6) [class_id, xmin, ymin, width, height, percent_contained]
    # Everything is in relative coordinates.
    # Coordinates of the square box containing the GT box:
    box_center_x = bbox[1] + bbox[3] / 2.0
    box_center_y = bbox[2] + bbox[4] / 2.0
    box_sq_side = max(bbox[3], bbox[4])
    box_sq_xmin = max(box_center_x - box_sq_side / 2.0, 0)
    box_sq_ymin = max(box_center_y - box_sq_side / 2.0, 0)
    box_sq_xmax = box_sq_xmin + box_sq_side
    box_sq_ymax = box_sq_ymin + box_sq_side
    if box_sq_xmax > 1:
        box_sq_xmax = 1
        box_sq_xmin = 1 - box_sq_side
    if box_sq_ymax > 1:
        box_sq_ymax = 1
        box_sq_ymin = 1 - box_sq_side

    # Maximum square patch centered on the GT box:
    margin_left = box_sq_xmin
    margin_right = 1 - box_sq_xmax
    margin_top = box_sq_ymin
    margin_bottom = 1 - box_sq_ymax
    min_margin_x = min(margin_left, margin_right)
    min_margin_y = min(margin_top, margin_bottom)
    max_width = 2 * min_margin_x + box_sq_side
    max_height = 2 * min_margin_y + box_sq_side
    max_patch_side = min(max_width, max_height)

    # Limit the patch side using the Area Ratio constrain:
    max_patch_side_by_ar = box_sq_side / np.sqrt(min_ar)
    max_patch_side = min(max_patch_side, max_patch_side_by_ar)
    min_patch_side = box_sq_side / np.sqrt(max_ar)
    min_patch_side = min(min_patch_side, max_patch_side)
    assert min_patch_side <= max_patch_side, 'min_patch_side > max_patch_side'

    # Sample patch side:
    patch_side = min_patch_side + np.random.rand() * (max_patch_side - min_patch_side)

    # Patch center:
    dc_max_side = max_dc / np.sqrt(2)
    dc_x = np.random.rand() * dc_max_side
    dc_y = np.random.rand() * dc_max_side
    patch_center_x = box_center_x - patch_side * dc_x / 2.0
    patch_center_y = box_center_y - patch_side * dc_y / 2.0

    # Patch final coordinates:
    patch_xmin = max(patch_center_x - patch_side / 2.0, 0)
    patch_ymin = max(patch_center_y - patch_side / 2.0, 0)
    if patch_xmin + patch_side > 1:
        patch_side = 1 - patch_xmin
    if patch_ymin + patch_side > 1:
        patch_side = 1 - patch_ymin
    return patch_xmin, patch_ymin, patch_side


def sample_random_patch(image, bboxes, img_side, min_scale, max_scale, padding_factor):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    patch_x0_rel, patch_y0_rel, patch_side_rel = make_patch_shape(min_scale, max_scale, padding_factor)
    patch, remaining_boxes = sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_side_rel)
    # patch: (patch_side_abs, patch_side_abs, 3)
    # remaining_boxes: (n_remaining_boxes, 9) [class_id, xmin, ymin, width, height, percent_contained, gt_idx, c_x_unclipped, c_y_unclipped]
    return patch, remaining_boxes


def sample_patch(image, bboxes, img_side, patch_x0_rel, patch_y0_rel, patch_side_rel):
    # image: (img_side, img_side, 3)
    # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_side).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_side).astype(np.int32), 0)
    patch_side_abs = np.round(patch_side_rel * img_side).astype(np.int32)
    patch_side_abs = max(min(patch_side_abs, img_side - patch_y0_abs), 1)
    # Image:
    patch = image[patch_y0_abs:(patch_y0_abs+patch_side_abs), patch_x0_abs:(patch_x0_abs+patch_side_abs), :]  # (patch_side_abs, patch_side_abs, 3)
    # Percent contained:
    pc = compute_pc(bboxes, [patch_x0_rel, patch_y0_rel, patch_side_rel, patch_side_rel])  # (n_gt)

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

    remaining_boxes_x0 = (remaining_boxes_on_orig_coords[:, 1] - patch_x0_rel) / patch_side_rel
    remaining_boxes_y0 = (remaining_boxes_on_orig_coords[:, 2] - patch_y0_rel) / patch_side_rel
    remaining_boxes_x1 = (remaining_boxes_on_orig_coords[:, 1] + remaining_boxes_on_orig_coords[:, 3] - patch_x0_rel) / patch_side_rel
    remaining_boxes_y1 = (remaining_boxes_on_orig_coords[:, 2] + remaining_boxes_on_orig_coords[:, 4] - patch_y0_rel) / patch_side_rel
    remaining_boxes_x_center = (remaining_boxes_x0 + remaining_boxes_x1) / 2.0
    remaining_boxes_y_center = (remaining_boxes_y0 + remaining_boxes_y1) / 2.0
    remaining_boxes_x0 = np.maximum(remaining_boxes_x0, 0)  # (n_remaining_boxes)
    remaining_boxes_y0 = np.maximum(remaining_boxes_y0, 0)  # (n_remaining_boxes)
    remaining_boxes_x1 = np.minimum(remaining_boxes_x1, 1)
    remaining_boxes_y1 = np.minimum(remaining_boxes_y1, 1)
    remaining_boxes_width = remaining_boxes_x1 - remaining_boxes_x0  # (n_remaining_boxes)
    remaining_boxes_height = remaining_boxes_y1 - remaining_boxes_y0  # (n_remaining_boxes)

    remaining_boxes = np.stack([remaining_boxes_class_id, remaining_boxes_x0,
                                remaining_boxes_y0, remaining_boxes_width,
                                remaining_boxes_height, remaining_boxes_pc,
                                remaining_boxes_gt_idx, remaining_boxes_x_center,
                                remaining_boxes_y_center], axis=-1)  # (n_remaining_boxes, 9)
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


def make_patch_shape(min_scale, max_scale, padding_factor):
    orig_side_rel = 1.0 / (1 + 2 * padding_factor)
    margin_rel = padding_factor * orig_side_rel
    assert np.abs(orig_side_rel + 2 * margin_rel - 1) < 1e-6, 'Error computing margin'
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    patch_side = scale
    min_c = max(margin_rel, scale / 2.0)
    max_c = min(1.0 - margin_rel, 1 - scale / 2.0)
    xc = np.random.rand() * (max_c - min_c) + min_c
    yc = np.random.rand() * (max_c - min_c) + min_c
    x0 = xc - scale / 2.0
    y0 = yc - scale / 2.0
    assert xc > margin_rel, 'xc <= margin_rel'
    assert xc < 1 - margin_rel, 'xc >= 1 - margin_rel'
    assert yc > margin_rel, 'yc <= margin_rel'
    assert yc < 1 - margin_rel, 'yc >= 1 - margin_rel'
    return x0, y0, patch_side




