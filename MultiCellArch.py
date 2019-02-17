import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from BoundingBoxes import BoundingBox, PredictedBox
import operator
import os
import cv2
import tools
import sys
import scipy
import network
from GridLevel import GridLevel
import CommonEncoding


class MultiCellArch:
    def __init__(self, options, nclasses, outdir, th_conf, classnames):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.n_grids = len(self.opts.grid_levels_size_pad)
        self.grid_levels = []
        for i in range(self.n_grids):
            size_pad = self.opts.grid_levels_size_pad[i]
            self.grid_levels.append(GridLevel(size_pad[0], size_pad[1]))
        self.grid_levels.sort(key=operator.attrgetter('input_size'))
        self.n_boxes = -1
        self.max_pad_rel = -1
        self.max_pad_abs = -1
        self.input_image_size_w_pad = -1
        self.expected_n_boxes = self.get_expected_num_boxes()
        self.encoded_gt_shape = (self.expected_n_boxes, 9)
        self.n_comparisons = -1
        self.metric_names = ['mAP']
        self.n_metrics = len(self.metric_names)
        # self.make_comparison_op()
        self.batch_count_debug = 0
        self.th_conf = th_conf
        self.classnames = classnames
        if self.opts.debug:
            self.debug_dir = os.path.join(outdir, 'debug')
            os.makedirs(self.debug_dir)

    def make_comparison_op(self):
        self.CR1 = tf.placeholder(shape=(self.opts.lcr), dtype=tf.float32)
        self.CR2 = tf.placeholder(shape=(self.opts.lcr), dtype=tf.float32)
        CRs = tf.stack([self.CR1, self.CR2], axis=-1)  # (lcr, 2)
        CRs = tf.expand_dims(CRs, axis=0)  # (1, lcr, 2)
        self.comparison_op = network.comparison(CRs, self.opts.lcr)  # (1, 2)
        self.comparison_op = tf.squeeze(self.comparison_op)  # (2)
        softmax = tf.nn.softmax(self.comparison_op, axis=-1)  # (2)
        self.pseudo_distance = softmax[0]  # ()

    def get_expected_num_boxes(self):
        expected_n_boxes = 0
        for grid in self.grid_levels:
            dim_this_grid = 1 + (grid.input_size_w_pad - network.receptive_field_size) // self.opts.step_in_pixels
            expected_n_boxes += dim_this_grid * dim_this_grid
        return expected_n_boxes

    def get_input_shape(self):
        input_shape = [self.opts.input_image_size, self.opts.input_image_size]
        return input_shape

    def compute_anchors_coordinates(self):
        self.anchors_coordinates = np.zeros(shape=(self.n_boxes, 4), dtype=np.float32)
        for pos in range(self.n_boxes):
            xmin, ymin, xmax, ymax = self.get_anchor_coords_wrt_padded(pos)
            self.anchors_coordinates[pos, 0] = xmin
            self.anchors_coordinates[pos, 1] = ymin
            self.anchors_coordinates[pos, 2] = xmax
            self.anchors_coordinates[pos, 3] = ymax
        # Transform to [x_center, y_center, width, height] coordinates:
        self.anchors_xc = (self.anchors_coordinates[:, 0] + self.anchors_coordinates[:, 2]) / 2.0
        self.anchors_yc = (self.anchors_coordinates[:, 1] + self.anchors_coordinates[:, 3]) / 2.0
        self.anchors_w = self.anchors_coordinates[:, 2] - self.anchors_coordinates[:, 0]
        self.anchors_h = self.anchors_coordinates[:, 3] - self.anchors_coordinates[:, 1]

    def make_input_multiscale(self, inputs):
        inputs_all_sizes = []
        for grid in self.grid_levels:
            size = grid.input_size
            if size == self.opts.input_image_size:
                inputs_this_grid = inputs
            else:
                inputs_this_grid = tf.image.resize_images(inputs, [size, size])
            if grid.pad_abs > 0:
                paddings = tf.constant([[0, 0], [grid.pad_abs, grid.pad_abs], [grid.pad_abs, grid.pad_abs], [0, 0]], name='paddings')
                inputs_this_grid = tf.pad(inputs_this_grid, paddings, name='pad_image')
            inputs_all_sizes.append(inputs_this_grid)
        return inputs_all_sizes

    def box_coords_wrt_padded_2_box_coords_wrt_anchor(self, box_coords_wrt_pad, position, convert_to_abs=False):
        box_xmin_wrt_pad = box_coords_wrt_pad[0]
        box_ymin_wrt_pad = box_coords_wrt_pad[1]
        box_width_wrt_pad = box_coords_wrt_pad[2]
        box_height_wrt_pad = box_coords_wrt_pad[3]
        anchor_xmin_wrt_pad, anchor_ymin_wrt_pad, anchor_xmax_wrt_pad, anchor_ymax_wrt_pad = self.get_anchor_coords_wrt_padded(position)
        anchor_width_wrt_pad = anchor_xmax_wrt_pad - anchor_xmin_wrt_pad
        anchor_height_wrt_pad = anchor_ymax_wrt_pad - anchor_ymin_wrt_pad
        box_xmin_wrt_anchor = (box_xmin_wrt_pad - anchor_xmin_wrt_pad) / anchor_width_wrt_pad
        box_ymin_wrt_anchor = (box_ymin_wrt_pad - anchor_ymin_wrt_pad) / anchor_height_wrt_pad
        box_width_wrt_anchor = box_width_wrt_pad / anchor_width_wrt_pad
        box_height_wrt_anchor = box_height_wrt_pad / anchor_height_wrt_pad
        box_coords_wrt_anchor = np.array([box_xmin_wrt_anchor, box_ymin_wrt_anchor, box_width_wrt_anchor, box_height_wrt_anchor], dtype=np.float32)
        if convert_to_abs:
            box_coords_wrt_anchor = np.clip(np.round(box_coords_wrt_anchor * network.receptive_field_size).astype(np.int32), 0, network.receptive_field_size - 1)
        return box_coords_wrt_anchor

    def net_on_every_size(self, inputs_all_sizes):
        self.n_boxes = 0
        all_outputs = []
        all_crs = []
        for i in range(len(self.grid_levels)):
            grid = self.grid_levels[i]
            print('')
            print('Defining network for input size ' + str(grid.input_size) + ' (pad: ' + str(grid.pad_abs) + ')')
            common_representation = network.common_representation(inputs_all_sizes[i], self.opts.lcr)
            net = network.localization_and_classification_path(common_representation, self.opts, self.nclasses)
            net_shape = net.shape.as_list()
            assert net_shape[1] == net_shape[2], 'Different width and height at network output'
            grid.set_output_shape(net_shape[1], network.receptive_field_size)
            grid.set_flat_start_pos(self.n_boxes)
            self.max_pad_rel = max(self.max_pad_rel, grid.pad_rel)
            output_flat = tf.reshape(net, [-1, grid.n_boxes, 4+self.nclasses], name='output_flat')
            cr_flat = tf.reshape(common_representation, [-1, grid.n_boxes, self.opts.lcr], name='cr_flat')
            self.n_boxes += grid.n_boxes
            all_outputs.append(output_flat)
            all_crs.append(cr_flat)
            if (grid.input_size_w_pad - network.receptive_field_size) / self.opts.step_in_pixels + 1 != grid.output_shape:
                raise Exception('Inconsistent step for input size ' + str(grid.input_size_w_pad) + '. Grid size is ' + str(grid.output_shape) + '.')
        assert self.n_boxes == self.get_expected_num_boxes(), 'Expected number of boxes differs from the real number.'
        all_outputs = tf.concat(all_outputs, axis=1, name='all_outputs')  # (batch_size, nboxes, 4+nclasses)
        all_crs = tf.concat(all_crs, axis=1, name='all_crs')  # (batch_size, nboxes, lcr)
        self.input_image_size_w_pad = int(np.ceil(float(self.opts.input_image_size) / (1 - 2 * self.max_pad_rel)))
        self.max_pad_abs = int(np.round(self.max_pad_rel * self.input_image_size_w_pad))
        self.input_image_size_w_pad = self.opts.input_image_size + 2 * self.max_pad_abs
        print('')
        print('Maximum relative pad: ' + str(self.max_pad_rel))
        print('Maximum absolute pad: ' + str(self.max_pad_abs))
        print('Input image size with pad: ' + str(self.input_image_size_w_pad))
        self.print_grids_info()
        return all_outputs, all_crs  # (batch_size, nboxes, ?)


    def print_grids_info(self):
        for i in range(len(self.grid_levels)):
            grid = self.grid_levels[i]
            print('')
            print('Grid ' + str(i))
            print('Image size: ' + str(grid.input_size))
            print('Padded image size: ' + str(grid.input_size_w_pad))
            print('Absolute pad: ' + str(grid.pad_abs))
            print('Relative pad: ' + str(grid.pad_rel))
            print('Relative box side: ' + str(grid.rel_box_size))
            print('Number of boxes in each dimension: ' + str(grid.output_shape))
            print('Number of boxes: ' + str(grid.n_boxes))
            print('Start position in flat representation: ' + str(grid.flat_start_pos))
            min_object_area = int(np.round(self.input_image_size_w_pad * np.sqrt(self.opts.threshold_ar) * grid.rel_box_size))
            print('Minimum object size that this grid can detect: ' + str(min_object_area) + 'x' + str(min_object_area))
            if i < len(self.grid_levels) - 1:
                print('Area ratio with next grid level: ' +
                      str(float(self.grid_levels[i + 1].rel_box_size * self.grid_levels[i + 1].rel_box_size) /
                          (grid.rel_box_size * grid.rel_box_size)))
        print('')
        print('Total number of boxes: ' + str(self.n_boxes))

    def make(self, inputs):
        inputs_all_sizes = self.make_input_multiscale(inputs)
        net_output, CRs = self.net_on_every_size(inputs_all_sizes)  # (batch_size, nboxes, ?)
        self.compute_anchors_coordinates()
        if self.opts.debug:
            debug_inputs = [net_output]
            debug_inputs.extend(inputs_all_sizes)
            net_output_shape = net_output.shape
            net_output = tf.py_func(self.write_debug_info, debug_inputs, (tf.float32))
            net_output.set_shape(net_output_shape)
        localizations, softmax = self.obtain_localizations_and_softmax(net_output)
        return localizations, softmax, CRs

    def obtain_localizations_and_softmax(self, net_output):
        # net_output: (batch_size, nboxes, 4+nclasses)
        localizations_enc = net_output[:, :, :4]  # (batch_size, nboxes, 4)
        localizations_dec = self.decode_boxes_tf(localizations_enc)  # (batch_size, nboxes, 4)
        logits = net_output[:, :, 4:]  # (batch_size, nboxes, nclasses)
        softmax = tf.nn.softmax(logits, axis=-1)  # (batch_size, nboxes, nclasses)
        return localizations_dec, softmax


    def get_grid_idx_from_flat_position(self, position):
        limit = 0
        for i in range(self.n_grids):
            limit += self.grid_levels[i].n_boxes
            if position < limit:
                return i
        return -1

    def get_grid_coordinates_from_flat_position(self, position):
        grid_idx = self.get_grid_idx_from_flat_position(position)
        grid = self.grid_levels[grid_idx]
        position_in_grid = position - grid.flat_start_pos
        row = position_in_grid // grid.output_shape
        col = position_in_grid % grid.output_shape
        return grid_idx, row, col

    def get_anchor_coords_wrt_its_input(self, position, make_absolute=False):
        grid_idx, row, col = self.get_grid_coordinates_from_flat_position(position)
        grid = self.grid_levels[grid_idx]
        # Relative coordinates:
        xmin = float(self.opts.step_in_pixels * col) / grid.input_size_w_pad
        ymin = float(self.opts.step_in_pixels * row) / grid.input_size_w_pad
        # xmin = float(self.opts.step_in_pixels * row) / grid.input_size_w_pad
        # ymin = float(self.opts.step_in_pixels * col) / grid.input_size_w_pad
        xmax = xmin + grid.rel_box_size
        ymax = ymin + grid.rel_box_size
        if make_absolute:
            # Absolute coordinates:
            xmin = max(int(np.round(xmin * grid.input_size_w_pad)), 0)
            ymin = max(int(np.round(ymin * grid.input_size_w_pad)), 0)
            xmax = min(int(np.round(xmax * grid.input_size_w_pad)), grid.input_size_w_pad)
            ymax = min(int(np.round(ymax * grid.input_size_w_pad)), grid.input_size_w_pad)
        return xmin, ymin, xmax, ymax

    # With respect to padded image.
    def get_anchor_coords_wrt_padded(self, position, make_absolute=False):
        # print('get_anchor_coords_wrt_padded')
        grid_idx = self.get_grid_idx_from_flat_position(position)
        grid = self.grid_levels[grid_idx]
        # Relative coordinates:
        xmin, ymin, xmax, ymax = self.get_anchor_coords_wrt_its_input(position, make_absolute=False)
        # print('anchor_coords_wrt_its_input = ' + str([xmin, ymin, xmax, ymax]))
        # Take into account padding:
        pad_dif = self.max_pad_rel - grid.pad_rel
        xmin = pad_dif + xmin * (1 - 2 * pad_dif)
        ymin = pad_dif + ymin * (1 - 2 * pad_dif)
        xmax = pad_dif + xmax * (1 - 2 * pad_dif)
        ymax = pad_dif + ymax * (1 - 2 * pad_dif)
        if make_absolute:
            xmin = max(int(np.round(xmin * self.input_image_size_w_pad)), 0)
            ymin = max(int(np.round(ymin * self.input_image_size_w_pad)), 0)
            xmax = min(int(np.round(xmax * self.input_image_size_w_pad)), self.input_image_size_w_pad)
            ymax = min(int(np.round(ymax * self.input_image_size_w_pad)), self.input_image_size_w_pad)
        return xmin, ymin, xmax, ymax

    def remove_padding_from_gt(self, gt_boxes, clip=True):
        unpadded_gt_boxes = []
        for box in gt_boxes:
            unpadded_gt_boxes.append(box)
        for box in unpadded_gt_boxes:
            box.remove_padding(self.max_pad_rel, clip)
        return unpadded_gt_boxes

    def coords_orig2pad(self, coords_orig):
        coords_pad = np.zeros(shape=coords_orig.shape, dtype=np.float32)
        coords_pad[..., :2] = self.max_pad_rel + coords_orig[..., :2] * (1 - 2 * self.max_pad_rel)
        coords_pad[..., 2:] = coords_orig[..., 2:] * (1 - 2 * self.max_pad_rel)
        return coords_pad

    def encode_gt(self, gt_boxes):
        # Inputs:
        #     gt_boxes: List of GroundTruthBox objects, with relative coordinates (between 0 and 1)
        # Outputs:
        #     encoded_labels: numpy array of dimension nboxes x 9, being nboxes the total number of anchor boxes.
        #                     The second dimension is like follows: First the 4 encoded coordinates of the bounding box,
        #                     then a flag indicating if it matches a ground truth box or not, another flag inidicating
        #                     if it is a neutral box, the class id, the ground truth box id associated with, and then
        #                     the maximum percent contained (or the percent contained of the associated gt box)
        #
        # We don't consider the batch dimension here.
        n_gt = len(gt_boxes)

        if n_gt > 0:
            gt_vec = np.zeros(shape=(n_gt, 4), dtype=np.float32)
            gt_class_ids = np.zeros(shape=(n_gt), dtype=np.int32)
            gt_pc_incoming = np.zeros(shape=(n_gt), dtype=np.float32)
            for gt_idx in range(n_gt):
                gt_vec[gt_idx, :] = gt_boxes[gt_idx].get_coords()
                # Take into account padding:
                gt_class_ids[gt_idx] = gt_boxes[gt_idx].classid
                gt_pc_incoming[gt_idx] = gt_boxes[gt_idx].pc

            # Convert bounding boxes coordinates from the original representation, to the representation in the padded image:
            gt_vec = self.coords_orig2pad(gt_vec)

            pc, ar, dc = self.compute_pc_ar_dc(gt_vec)  # All variables have shape (n_gt, n_boxes)

            gt_pc_incoming_exp = np.expand_dims(gt_pc_incoming, axis=1)  # (n_gt, 1)
            gt_pc_incoming_exp = np.tile(gt_pc_incoming_exp, [1, self.n_boxes])  # (n_gt, n_boxes)
            pc = pc * gt_pc_incoming_exp  # (n_gt, n_boxes)

            # Positive boxes:
            mask_ar = ar > self.opts.threshold_ar  # (n_gt, n_boxes)
            mask_pc = pc > self.opts.threshold_pc  # (n_gt, n_boxes)
            mask_dc = dc < self.opts.threshold_dc  # (n_gt, n_boxes)
            mask_thresholds = mask_ar & mask_pc & mask_dc  # (n_gt, n_boxes)
            mask_match = np.any(mask_thresholds, axis=0)  # (n_boxes)

            dc_masked = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt, self.n_boxes), dtype=np.float32) ) # (n_gt, n_boxes)

            nearest_valid_gt_idx = np.argmin(dc_masked, axis=0)  # (n_boxes)

            # Neutral boxes:
            mask_ar_neutral = ar > self.opts.threshold_ar_neutral  # (n_gt, n_boxes)
            mask_pc_neutral = pc > self.opts.threshold_pc_neutral  # (n_gt, n_boxes)
            mask_dc_neutral = dc < self.opts.threshold_dc_neutral  # (n_gt, n_boxes)
            mask_thresholds_neutral = mask_ar_neutral & mask_pc_neutral & mask_dc_neutral  # (n_gt, n_boxes)
            mask_neutral = np.any(mask_thresholds_neutral, axis=0)  # (n_boxes)
            mask_neutral = np.logical_and(mask_neutral, np.logical_not(mask_match))  # (n_boxes)

            # Get the coordinates and the class id of the gt boxes matched:
            coordinates = np.take(gt_vec, nearest_valid_gt_idx, axis=0)  # (n_boxes, 4)
            coordinates_enc = self.encode_boxes(coordinates)  # (nboxes, 4)
            class_ids_pos = np.take(gt_class_ids, nearest_valid_gt_idx, axis=0)  # (n_boxes)

            # Negative boxes:
            mask_negative = np.logical_and(np.logical_not(mask_match), np.logical_not(mask_neutral))  # (n_boxes)
            background_ids = np.ones(shape=(self.n_boxes), dtype=np.int32) * self.background_id
            class_ids = np.where(mask_negative, background_ids, class_ids_pos)  # (n_boxes)

            # Percent contained associated with each anchor box.
            # This is the PC of the assigned gt box, if there is any, or otherwise the maximum PC it has.
            pc_associated = np.zeros(shape=(self.n_boxes), dtype=np.float32)
            for i in range(self.n_boxes):
                if mask_match[i]:
                    pc_associated[i] = pc[nearest_valid_gt_idx[i], i]
                else:
                    pc_associated[i] = np.max(pc[:, i])

            # Put all together in one array:
            labels_enc = coordinates_enc  # (nboxes, 4)
            labels_enc = np.concatenate((labels_enc, np.expand_dims(mask_match.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 5)  pos 4
            labels_enc = np.concatenate((labels_enc, np.expand_dims(mask_neutral.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 6)  pos 5
            labels_enc = np.concatenate((labels_enc, np.expand_dims(class_ids.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 7)  pos 6
            labels_enc = np.concatenate((labels_enc, np.expand_dims(nearest_valid_gt_idx.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 8)  pos 7
            labels_enc = np.concatenate((labels_enc, np.expand_dims(pc_associated, axis=1)), axis=1)  # (n_boxes, 9)  pos 8

        else:
            labels_enc = np.zeros(shape=(self.n_boxes, 9), dtype=np.float32)
            labels_enc[:, 6] = self.background_id

        return labels_enc

    def decode_gt(self, labels_enc, remove_duplicated=True):
        # Inputs:
        #     labels_enc: numpy array of dimension (nboxes x 9).
        #     remove_duplicated: Many default boxes can be assigned to the same ground truth box. Therefore,
        #                        when decoding, we can find several times the same box. If we set this to True, then
        #                        the dubplicated boxes are deleted.
        # Outputs:
        #     gt_boxes: List of BoundingBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.
        gt_boxes = []

        coords_enc = labels_enc[:, :4]  # (nboxes, 4)
        coords_raw = self.decode_boxes(coords_enc)  # (nboxes, 4)

        for anchor_idx in range(self.n_boxes):
            if labels_enc[anchor_idx, 4] > 0.5:
                classid = int(np.round(labels_enc[anchor_idx, 6]))
                # percent_contained = sigmoid(labels_enc[anchor_idx, 8])
                percent_contained = np.clip(labels_enc[anchor_idx, 8], 0.0, 1.0)
                gtbox = BoundingBox(coords_raw[anchor_idx, :], classid, percent_contained)
                # Skip if it is duplicated:
                if remove_duplicated:
                    if not self.check_duplicated(gt_boxes, gtbox):
                        gt_boxes.append(gtbox)
                else:
                    gt_boxes.append(gtbox)
        # Convert boxes coordinates from padded representation to the original one, and clip values to (0, 1):
        gt_boxes = self.remove_padding_from_gt(gt_boxes, clip=True)
        return gt_boxes

    def decode_preds(self, net_output_nobatch, th_conf):
        # Inputs:
        #     net_output_nobatch: numpy array of dimension nboxes x 9.
        # Outputs:
        #     predictions: List of PredictedBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.

        net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(net_output_nobatch, self.opts, self.nclasses)
        # net_output_conf: (nboxes, nclasses)
        # net_output_coords: (nboxes, ?)
        # net_output_pc: (nboxes, ?)
        if self.opts.box_per_class:
            # net_output_coords: (nboxes, 4 * nclasses)
            selected_coords = CommonEncoding.get_selected_coords_np(net_output_conf, net_output_coords, self.nclasses, self.opts.box_per_class)  # (nboxes, 4)
            pred_coords_raw = self.decode_boxes(selected_coords)  # (nboxes, 4) [xmin, ymin, width, height]
        else:
            # net_output_coords: (nboxes, 4)
            pred_coords_raw = self.decode_boxes(net_output_coords)  # (nboxes, 4) [xmin, ymin, width, height]
        class_ids = np.argmax(net_output_conf, axis=1)  # (nboxes)
        predictions = []
        for idx in range(self.n_boxes):
            cls_id = class_ids[idx]
            if cls_id != self.background_id:
                gtbox = PredictedBox(pred_coords_raw[idx], cls_id, net_output_conf[idx, cls_id])
                if self.opts.predict_pc:
                    if self.opts.box_per_class:
                        # net_output_pc: (nboxes, nclasses)
                        gtbox.pc = net_output_pc[idx, cls_id]
                    else:
                        # net_output_pc: (nboxes)
                        gtbox.pc = net_output_pc[idx]
                predictions.append(gtbox)
        # Convert boxes coordinates from padded representation to the original one, and clip values to (0, 1):
        predictions = self.remove_padding_from_gt(predictions, clip=True)
        return predictions

    def encode_boxes(self, coords_raw):
        # coords_raw: (nboxes, 4)
        xmin = coords_raw[:, 0]
        ymin = coords_raw[:, 1]
        width = coords_raw[:, 2]
        height = coords_raw[:, 3]

        xc = xmin + 0.5 * width
        yc = ymin + 0.5 * height

        dcx = (self.anchors_xc - xc) / (self.anchors_w * 0.5)  # (nboxes)
        dcy = (self.anchors_yc - yc) / (self.anchors_h * 0.5)  # (nboxes)
        # Between -1 and 1 for the box to lie inside the anchor.

        w_rel = width / self.anchors_w  # (nboxes)
        h_rel = height / self.anchors_h  # (nboxes)
        # Between 0 and 1 for the box to lie inside the anchor.

        # Encoding step:
        if self.opts.encoding_method == 'basic_1':
            dcx_enc = np.tan(dcx * (np.pi / 2.0 - self.opts.enc_epsilon))
            dcy_enc = np.tan(dcy * (np.pi / 2.0 - self.opts.enc_epsilon))
            w_enc = CommonEncoding.logit((w_rel - self.opts.enc_wh_b) / self.opts.enc_wh_a)
            h_enc = CommonEncoding.logit((h_rel - self.opts.enc_wh_b) / self.opts.enc_wh_a)
        elif self.opts.encoding_method == 'ssd':
            dcx_enc = dcx * 10.0
            dcy_enc = dcy * 10.0
            w_enc = np.log(w_rel) * 5.0
            h_enc = np.log(h_rel) * 5.0
        elif self.opts.encoding_method == 'no_encode':
            dcx_enc = dcx
            dcy_enc = dcy
            w_enc = w_rel
            h_enc = h_rel
        else:
            raise Exception('Encoding method not recognized.')

        coords_enc = np.stack([dcx_enc, dcy_enc, w_enc, h_enc], axis=1)  # (nboxes, 4)

        return coords_enc  # (nboxes, 4) [dcx_enc, dcy_enc, w_enc, h_enc]

    def decode_boxes_tf(self, coords_enc):
        # coords_enc: (..., 4)
        dcx_enc = coords_enc[..., 0]
        dcy_enc = coords_enc[..., 1]
        w_enc = coords_enc[..., 2]
        h_enc = coords_enc[..., 3]

        # Decoding step:
        if self.opts.encoding_method == 'basic_1':
            dcx_rel = tf.clip_by_value(tf.atan(dcx_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)
            dcy_rel = tf.clip_by_value(tf.atan(dcy_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)
            w_rel = tf.clip_by_value(CommonEncoding.sigmoid(w_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
            h_rel = tf.clip_by_value(CommonEncoding.sigmoid(h_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'ssd':
            dcx_rel = tf.clip_by_value(dcx_enc * 0.1, -1.0, 1.0)
            dcy_rel = tf.clip_by_value(dcy_enc * 0.1, -1.0, 1.0)
            w_rel = tf.clip_by_value(tf.exp(w_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
            h_rel = tf.clip_by_value(tf.exp(h_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'no_encode':
            dcx_rel = tf.clip_by_value(dcx_enc, -1.0, 1.0)
            dcy_rel = tf.clip_by_value(dcy_enc, -1.0, 1.0)
            w_rel = tf.clip_by_value(w_enc, 1.0 / network.receptive_field_size, 1.0)
            h_rel = tf.clip_by_value(h_enc, 1.0 / network.receptive_field_size, 1.0)
        else:
            raise Exception('Encoding method not recognized.')

        xc = self.anchors_xc - dcx_rel * (self.anchors_w * 0.5)  # (nboxes)
        yc = self.anchors_yc - dcy_rel * (self.anchors_h * 0.5)  # (nboxes)

        width = self.anchors_w * w_rel  # (nboxes)
        height = self.anchors_h * h_rel  # (nboxes)

        xmin = xc - 0.5 * width  # (nboxes)
        ymin = yc - 0.5 * height  # (nboxes)


        # Remove padding:
        xmin = (xmin - self.max_pad_rel) / (1 - 2 * self.max_pad_rel)
        ymin = (ymin - self.max_pad_rel) / (1 - 2 * self.max_pad_rel)
        width = width / (1 - 2 * self.max_pad_rel)
        height = height / (1 - 2 * self.max_pad_rel)
        # Clip:
        xmin = tf.clip_by_value(xmin, 0.0, 1.0)
        ymin = tf.clip_by_value(ymin, 0.0, 1.0)
        width = tf.maximum(0.0, tf.minimum(1.0 - xmin, width))
        height = tf.maximum(0.0, tf.minimum(1.0 - ymin, height))




        coords_raw = tf.stack([xmin, ymin, width, height], axis=-1)  # (nboxes, 4)

        return coords_raw  # (..., 4) [xmin, ymin, width, height]

    def decode_boxes(self, coords_enc):
        # coords_enc: (..., 4)
        dcx_enc = coords_enc[..., 0]
        dcy_enc = coords_enc[..., 1]
        w_enc = coords_enc[..., 2]
        h_enc = coords_enc[..., 3]

        # Decoding step:
        if self.opts.encoding_method == 'basic_1':
            dcx_rel = np.clip(np.arctan(dcx_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)
            dcy_rel = np.clip(np.arctan(dcy_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)
            w_rel = np.clip(CommonEncoding.sigmoid(w_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
            h_rel = np.clip(CommonEncoding.sigmoid(h_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'ssd':
            dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)
            w_rel = np.clip(np.exp(w_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
            h_rel = np.clip(np.exp(h_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'no_encode':
            dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
            w_rel = np.clip(w_enc, 1.0 / network.receptive_field_size, 1.0)
            h_rel = np.clip(h_enc, 1.0 / network.receptive_field_size, 1.0)
        else:
            raise Exception('Encoding method not recognized.')

        xc = self.anchors_xc - dcx_rel * (self.anchors_w * 0.5)  # (nboxes)
        yc = self.anchors_yc - dcy_rel * (self.anchors_h * 0.5)  # (nboxes)

        width = self.anchors_w * w_rel  # (nboxes)
        height = self.anchors_h * h_rel  # (nboxes)

        xmin = xc - 0.5 * width  # (nboxes)
        ymin = yc - 0.5 * height  # (nboxes)

        # # # Ensure we don't have a degenerate box:
        # xmin = np.minimum(np.maximum(xmin, 0), anchors_w - 2)
        # ymin = np.minimum(np.maximum(ymin, 0), anchors_h - 2)
        # width = np.minimum(np.maximum(width, 1), anchors_w - xmin - 1)
        # height = np.minimum(np.maximum(height, 1), anchors_h - ymin - 1)

        coords_raw = np.stack([xmin, ymin, width, height], axis=-1)  # (nboxes, 4)

        return coords_raw  # (..., 4) [xmin, ymin, width, height]

    def decode_boxes_w_batch_size(self, coords_enc):
        # coords_enc: (batch_size, nboxes, 4)
        dcx_enc = coords_enc[:, :, 0]  # (batch_size, nboxes)
        dcy_enc = coords_enc[:, :, 1]  # (batch_size, nboxes)
        w_enc = coords_enc[:, :, 2]  # (batch_size, nboxes)
        h_enc = coords_enc[:, :, 3]  # (batch_size, nboxes)

        # Decoding step:
        if self.opts.encoding_method == 'basic_1':
            dcx_rel = np.clip(np.arctan(dcx_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)  # (batch_size, nboxes)
            dcy_rel = np.clip(np.arctan(dcy_enc) / (np.pi / 2.0 - self.opts.enc_epsilon), -1.0, 1.0)  # (batch_size, nboxes)
            w_rel = np.clip(CommonEncoding.sigmoid(w_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)  # (batch_size, nboxes)
            h_rel = np.clip(CommonEncoding.sigmoid(h_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)  # (batch_size, nboxes)
        elif self.opts.encoding_method == 'ssd':
            dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)  # (batch_size, nboxes)
            dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)  # (batch_size, nboxes)
            w_rel = np.clip(np.exp(w_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)  # (batch_size, nboxes)
            h_rel = np.clip(np.exp(h_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)  # (batch_size, nboxes)
        elif self.opts.encoding_method == 'no_encode':
            dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
            w_rel = np.clip(w_enc, 1.0 / network.receptive_field_size, 1.0)
            h_rel = np.clip(h_enc, 1.0 / network.receptive_field_size, 1.0)
        else:
            raise Exception('Encoding method not recognized.')

        # self.anchors_coordinates (nboxes, 4) [xmin, ymin, xmax, ymax]
        anchors_coords_exp = np.expand_dims(self.anchors_coordinates, axis=0)  # (1, nboxes, 4)
        batch_size = coords_enc.shape[0]
        anchors_coords_exp = np.tile(anchors_coords_exp, [batch_size, 1, 1])  # (batch_size, nboxes, 4)
        anchors_xc_exp = (anchors_coords_exp[:, :, 0] + anchors_coords_exp[:, :, 2]) / 2.0  # (batch_size, nboxes)
        anchors_yc_exp = (anchors_coords_exp[:, :, 1] + anchors_coords_exp[:, :, 3]) / 2.0  # (batch_size, nboxes)
        anchors_w_exp = anchors_coords_exp[:, :, 2] - anchors_coords_exp[:, :, 0]  # (batch_size, nboxes)
        anchors_h_exp = anchors_coords_exp[:, :, 3] - anchors_coords_exp[:, :, 1]  # (batch_size, nboxes)

        xc = anchors_xc_exp - dcx_rel * (anchors_w_exp * 0.5)  # (batch_size, nboxes)
        yc = anchors_yc_exp - dcy_rel * (anchors_h_exp * 0.5)  # (batch_size, nboxes)

        width = anchors_w_exp * w_rel  # (batch_size, nboxes)
        height = anchors_h_exp * h_rel  # (batch_size, nboxes)

        xmin = xc - 0.5 * width  # (batch_size, nboxes)
        ymin = yc - 0.5 * height  # (batch_size, nboxes)

        # # # Ensure we don't have a degenerate box:
        # xmin = np.minimum(np.maximum(xmin, 0), anchors_w - 2)
        # ymin = np.minimum(np.maximum(ymin, 0), anchors_h - 2)
        # width = np.minimum(np.maximum(width, 1), anchors_w - xmin - 1)
        # height = np.minimum(np.maximum(height, 1), anchors_h - ymin - 1)

        coords_raw = np.stack([xmin, ymin, width, height], axis=-1)  # (batch_size, nboxes, 4)

        return coords_raw  # (batch_size, nboxes, 4) [xmin, ymin, width, height]

    def check_duplicated(self, all_boxes, new_box, tolerance=1e-6):
        for old_box in all_boxes:
            if old_box.classid == new_box.classid:
                old_coords = np.array(old_box.get_coords())
                new_coords = np.array(new_box.get_coords())
                if np.sum(np.square(old_coords - new_coords)) < tolerance:
                    return True
        return False

    # Compute the 'Percent Contained', the 'Area Ratio' and the 'Distance to Center' between ground truth boxes and and anchor boxes.
    def compute_pc_ar_dc(self, gt_boxes):
        # gt_boxes (n_gt, 4) [xmin, ymin, width, height]  # Parameterized with the top-left coordinates, and the width and height.
        # self.anchors_coordinates is parameterized with the corner coordinates (xmin, ymin, xmax, ymax)
        # Coordinates are relative (between 0 and 1)
        # We consider the square containing the ground truth box.

        n_gt = gt_boxes.shape[0]

        orig_boxes = np.stack([gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]], axis=1)
        orig_boxes_expanded = np.expand_dims(orig_boxes, axis=1)  # (n_gt, 1, 4)
        orig_boxes_expanded = np.tile(orig_boxes_expanded, (1, self.n_boxes, 1))  # (n_gt, nboxes, 4) [xmin, ymin, xmax, ymax]

        gt_boxes_xcenter = gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0
        gt_boxes_ycenter = gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0
        gt_boxes_maxside = np.maximum(gt_boxes[:, 2], gt_boxes[:, 3])
        gt_boxes_square_xmin = np.maximum(gt_boxes_xcenter - gt_boxes_maxside / 2.0, 0)
        gt_boxes_square_ymin = np.maximum(gt_boxes_ycenter - gt_boxes_maxside / 2.0, 0)
        gt_boxes_square_xmax = np.minimum(gt_boxes_xcenter + gt_boxes_maxside / 2.0, 1.0)
        gt_boxes_square_ymax = np.minimum(gt_boxes_ycenter + gt_boxes_maxside / 2.0, 1.0)
        gt_boxes_square = np.stack([gt_boxes_square_xmin, gt_boxes_square_ymin, gt_boxes_square_xmax, gt_boxes_square_ymax], axis=1)  # (n_gt, 4) [xmin, ymin, xmax, ymax]
        # Expand gt boxes:
        sq_boxes_expanded = np.expand_dims(gt_boxes_square, axis=1)  # (n_gt, 1, 4)
        sq_boxes_expanded = np.tile(sq_boxes_expanded, (1, self.n_boxes, 1))  # (n_gt, nboxes, 4)

        # Expand anchor boxes:
        anchors_expanded = np.expand_dims(self.anchors_coordinates, axis=0)  # (1, nboxes, 4)
        anchors_expanded = np.tile(anchors_expanded, (n_gt, 1, 1))  # (n_gt, nboxes, 4) [xmin, ymin, xmax, ymax]

        # Compute intersection:
        xmin = np.maximum(orig_boxes_expanded[:, :, 0], anchors_expanded[:, :, 0])  # (n_gt, nboxes)
        ymin = np.maximum(orig_boxes_expanded[:, :, 1], anchors_expanded[:, :, 1])  # (n_gt, nboxes)
        xmax = np.minimum(orig_boxes_expanded[:, :, 2], anchors_expanded[:, :, 2])  # (n_gt, nboxes)
        ymax = np.minimum(orig_boxes_expanded[:, :, 3], anchors_expanded[:, :, 3])  # (n_gt, nboxes)
        zero_grid = np.zeros((n_gt, self.n_boxes))  # (n_gt, nboxes)
        w = np.maximum(xmax - xmin, zero_grid)  # (n_gt, nboxes)
        h = np.maximum(ymax - ymin, zero_grid)  # (n_gt, nboxes)
        area_inter_orig = w * h  # (n_gt, nboxes)

        # Percent contained:
        area_gt_orig = (orig_boxes_expanded[:, :, 2] - orig_boxes_expanded[:, :, 0]) * (orig_boxes_expanded[:, :, 3] - orig_boxes_expanded[:, :, 1])  # (n_gt, nboxes)
        pc = area_inter_orig / area_gt_orig  # (n_gt, nboxes)

        # Area ratio:
        area_gt_sq = (sq_boxes_expanded[:, :, 2] - sq_boxes_expanded[:, :, 0]) * (sq_boxes_expanded[:, :, 3] - sq_boxes_expanded[:, :, 1])  # (n_gt, nboxes)
        area_anchors = (anchors_expanded[:, :, 2] - anchors_expanded[:, :, 0]) * (anchors_expanded[:, :, 3] - anchors_expanded[:, :, 1])  # (n_gt, nboxes)
        # ar = area_gt_orig / area_anchors  # (n_gt, nboxes)
        ar = area_gt_sq / area_anchors  #  (n_gt, nboxes)

        intersection_xcenter = (xmin + xmax) / 2.0  # (n_gt, nboxes)
        intersection_ycenter = (ymin + ymax) / 2.0  # (n_gt, nboxes)

        anchors_xcenter_expanded = (anchors_expanded[:, :, 0] + anchors_expanded[:, :, 2]) / 2.0  # (n_gt, nboxes)
        anchors_ycenter_expanded = (anchors_expanded[:, :, 1] + anchors_expanded[:, :, 3]) / 2.0  # (n_gt, nboxes)

        anchors_width_expanded = anchors_expanded[:, :, 2] - anchors_expanded[:, :, 0]
        anchors_height_expanded = anchors_expanded[:, :, 3] - anchors_expanded[:, :, 1]

        rel_distance_x = (anchors_xcenter_expanded - intersection_xcenter) / (anchors_width_expanded / 2.0)
        rel_distance_y = (anchors_ycenter_expanded - intersection_ycenter) / (anchors_height_expanded / 2.0)

        dc = np.sqrt(np.square(rel_distance_x) + np.square(rel_distance_y))  # (n_gt, nboxes)
        dc = np.minimum(dc, np.sqrt(2.0))

        return pc, ar, dc

    def write_debug_info(self, *arg):
        # arg: [net_output, inputs0, ..., inputsN]
        # net_output: (batch_size, nboxes, 4+nclasses)
        # inputsX: (batch_size, height, width, 3)
        net_output = arg[0]
        inputs_all_sizes = []
        for i in range(1, len(arg)):
            inputs_all_sizes.append(arg[i])
        batch_size = net_output.shape[0]
        localizations_enc = net_output[..., :4]  # (batch_size, nboxes, 4)
        logits = net_output[..., 4:]  # (batch_size, nboxes, nclasses)
        e_x = np.exp(logits - np.max(logits))  # (batch_size, nboxes, nclasses)
        denominator = np.tile(np.sum(e_x, axis=-1, keepdims=True), [1, 1, self.nclasses])  # (batch_size, nboxes, nclasses)
        softmax = e_x / denominator  # (batch_size, nboxes, nclasses)
        # softmax = tf.nn.softmax(logits, axis=-1)  # (batch_size, nboxes, nclasses)
        self.batch_count_debug += 1
        batch_dir = os.path.join(self.debug_dir, 'batch' + str(self.batch_count_debug))
        os.makedirs(batch_dir)
        for pos in range(self.n_boxes):
            grid_idx = self.get_grid_idx_from_flat_position(pos)
            inputs_this_box = inputs_all_sizes[grid_idx]  # (batch_size, height, width, 3)
            anchor_coords = self.get_anchor_coords_wrt_its_input(pos, True)  # [xmin, ymin, xmax, ymax]
            anc_xmin = anchor_coords[0]
            anc_ymin = anchor_coords[1]
            anc_xmax = anchor_coords[2]
            anc_ymax = anchor_coords[3]
            for img_idx in range(batch_size):
                img = inputs_this_box[img_idx, anc_ymin:anc_ymax, anc_xmin:anc_xmax].copy()  # (receptive_field_size, receptive_field_size, 3)
                img = tools.add_mean_again(img)
                path_to_save = os.path.join(batch_dir, 'img' + str(img_idx) + '_pos' + str(pos) + '.png')
                coords_enc = localizations_enc[img_idx, pos, :]  # (4)
                coords_dec = CommonEncoding.decode_boxes_np(coords_enc, self.opts)  # (4)
                if self.th_conf is None:
                    predicted_class = np.argmax(softmax[img_idx, pos, :])
                    if predicted_class != self.background_id:
                        conf = softmax[img_idx, pos, predicted_class]
                        bbox = np.concatenate([np.expand_dims(predicted_class, axis=0), coords_dec, np.expand_dims(conf, axis=0)], axis=0)  # (5)
                        bbox = np.expand_dims(bbox, axis=0)  # (1, 5)
                        img = tools.add_bounding_boxes_to_image2(img, bbox, self.classnames, color=(127, 0, 127))
                else:
                    predicted_class = np.argmax(softmax[img_idx, pos, :-1])
                    max_conf_no_bkg = softmax[img_idx, pos, predicted_class]
                    if max_conf_no_bkg > self.th_conf:
                        bbox = np.concatenate([np.expand_dims(predicted_class, axis=0), coords_dec, np.expand_dims(max_conf_no_bkg, axis=0)], axis=0)  # (5)
                        bbox = np.expand_dims(bbox, axis=0)  # (1, 5)
                        img = tools.add_bounding_boxes_to_image2(img, bbox, self.classnames, color=(127, 0, 127))
                cv2.imwrite(path_to_save, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        return net_output


def compute_iou_multi_dim(boxes1, boxes2):
    # boxes1: (..., 4) [xmin, ymin, width, height]
    # boxes2: (..., 4) [xmin, ymin, width, height]
    xmin = np.maximum(boxes1[..., 0], boxes2[..., 0])
    ymin = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xmax = np.minimum(boxes1[..., 0] + boxes1[..., 2], boxes2[..., 0] + boxes2[..., 2])
    ymax = np.minimum(boxes1[..., 1] + boxes1[..., 3], boxes2[..., 1] + boxes2[..., 3])
    intersection_area = np.maximum((xmax - xmin), 0.0) * np.maximum((ymax - ymin), 0.0)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = boxes1_area + boxes2_area - intersection_area
    iou = intersection_area / union_area
    return iou  # (...)




