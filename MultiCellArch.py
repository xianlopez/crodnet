import tensorflow as tf
import numpy as np
import operator
import os
import cv2
import tools
import network
from GridLevel import GridLevel
import CommonEncoding
import output_encoding
import gt_encoding


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
        self.input_image_size = 0
        for size_pad in self.opts.grid_levels_size_pad:
            self.input_image_size = max(self.input_image_size, size_pad[0])
        self.n_boxes = -1
        self.expected_n_boxes = self.get_expected_num_boxes()
        self.n_comparisons = -1
        self.metric_names = ['mAP']
        self.n_metrics = len(self.metric_names)
        self.batch_count_debug = 0
        self.th_conf = th_conf
        self.classnames = classnames
        self.n_labels = 8
        if self.opts.debug:
            self.debug_dir = os.path.join(outdir, 'debug')
            os.makedirs(self.debug_dir)

    def get_expected_num_boxes(self):
        expected_n_boxes = 0
        for grid in self.grid_levels:
            dim_this_grid = 1 + (grid.input_size_w_pad - network.receptive_field_size) // network.step_in_pixels
            expected_n_boxes += dim_this_grid * dim_this_grid
        return expected_n_boxes

    def get_input_shape(self):
        input_shape = [self.input_image_size, self.input_image_size]
        return input_shape

    def compute_anchors_coordinates(self):
        self.anchors_coordinates = np.zeros(shape=(self.n_boxes, 4), dtype=np.float32)
        for pos in range(self.n_boxes):
            xmin, ymin, xmax, ymax = self.get_anchor_coords_wrt_orig(pos)
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
            if size == self.input_image_size:
                inputs_this_grid = inputs
            else:
                inputs_this_grid = tf.image.resize_images(inputs, [size, size])
            if grid.pad_abs > 0:
                paddings = tf.constant([[0, 0], [grid.pad_abs, grid.pad_abs], [grid.pad_abs, grid.pad_abs], [0, 0]], name='paddings')
                inputs_this_grid = tf.pad(inputs_this_grid, paddings, name='pad_image')
            inputs_all_sizes.append(inputs_this_grid)
        return inputs_all_sizes

    def net_on_every_size(self, inputs_all_sizes):
        n_channels_last = CommonEncoding.get_n_channels_last(self.nclasses, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)
        self.n_boxes = 0
        all_outputs = []
        all_crs = []
        for i in range(len(self.grid_levels)):
            grid = self.grid_levels[i]
            print('')
            print('Defining network for input size ' + str(grid.input_size) + ' (pad: ' + str(grid.pad_abs) + ')')
            common_representation = network.common_representation(inputs_all_sizes[i], self.opts.lcr)
            net_output = network.prediction_path(common_representation, self.opts, n_channels_last)
            net_output_shape = net_output.shape.as_list()  # (batch_size, output_side, output_side, n_channels_last)
            assert net_output_shape[1] == net_output_shape[2], 'Different width and height at network output'
            grid.set_output_shape(net_output_shape[1], network.receptive_field_size)
            grid.set_flat_start_pos(self.n_boxes)
            output_flat = tf.reshape(net_output, [-1, grid.n_boxes, n_channels_last], name='output_flat')
            cr_flat = tf.reshape(common_representation, [-1, grid.n_boxes, self.opts.lcr], name='cr_flat')
            self.n_boxes += grid.n_boxes
            all_outputs.append(output_flat)
            all_crs.append(cr_flat)
            if (grid.input_size_w_pad - network.receptive_field_size) / network.step_in_pixels + 1 != grid.output_shape:
                raise Exception('Inconsistent step for input size ' + str(grid.input_size_w_pad) + '. Grid size is ' + str(grid.output_shape) + '.')
        assert self.n_boxes == self.get_expected_num_boxes(), 'Expected number of boxes differs from the real number.'
        all_outputs = tf.concat(all_outputs, axis=1, name='all_outputs')  # (batch_size, nboxes, n_channels_last)
        all_crs = tf.concat(all_crs, axis=1, name='all_crs')  # (batch_size, nboxes, lcr)
        self.print_grids_info()
        return all_outputs, all_crs


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
            anchor_ar = network.receptive_field_size * network.receptive_field_size / float(grid.input_size * grid.input_size)
            print('Anchor area ratio: ' + str(anchor_ar))
            min_object_rel_side = np.sqrt(anchor_ar * self.opts.threshold_ar_low)
            max_object_rel_side = np.sqrt(anchor_ar * self.opts.threshold_ar_high)
            print('Minimum object relative side to detect: ' + str(min_object_rel_side))
            print('Maximum object relative side to detect: ' + str(max_object_rel_side))
            if i < len(self.grid_levels) - 1:
                print('Area ratio with next grid level: ' +
                      str(float(self.grid_levels[i + 1].rel_box_size * self.grid_levels[i + 1].rel_box_size) /
                          (grid.rel_box_size * grid.rel_box_size)))
        print('')
        print('Total number of boxes: ' + str(self.n_boxes))

    def make_network(self, inputs):
        # inputs: (batch_size, input_image_size, input_image_size, 3)
        self.inputs_all_sizes = self.make_input_multiscale(inputs)
        self.net_output, CRs = self.net_on_every_size(self.inputs_all_sizes)  # (batch_size, nboxes, ?)
        self.compute_anchors_coordinates()

    def make_loss_metrics_and_preds(self, labels_enc, filenames):
        # labels_enc: (batch_size, nboxes, nlabels)
        self.compute_anchors_coordinates()
        # Make loss and metrics:
        loss, metrics = self.make_loss_and_metrics(self.net_output, labels_enc)  # ()
        if self.opts.debug:
            debug_inputs = [self.net_output]
            debug_inputs.extend(self.inputs_all_sizes)
            net_output_shape = self.net_output.shape
            net_output2 = tf.py_func(self.write_debug_info, debug_inputs, (tf.float32))
            net_output2.set_shape(net_output_shape)
            localizations, softmax = self.obtain_localizations_and_softmax(net_output2)
        else:
            localizations, softmax = self.obtain_localizations_and_softmax(self.net_output)
        return loss, metrics, localizations, softmax

    def obtain_localizations_and_softmax(self, net_output):
        # net_output: (batch_size, nboxes, ?)

        # Split net output:
        locs_enc = output_encoding.get_loc_enc(net_output)  # (batch_size, nboxes, 4)
        logits = output_encoding.get_logits(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes, nclasses)

        # Decode
        localizations_dec = self.decode_boxes_wrt_orig_tf(locs_enc)  # (batch_size, nboxes, 4)
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
        xmin = float(network.step_in_pixels * col) / grid.input_size_w_pad
        ymin = float(network.step_in_pixels * row) / grid.input_size_w_pad
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
    def get_anchor_coords_wrt_orig(self, position):
        # print('get_anchor_coords_wrt_padded')
        grid_idx = self.get_grid_idx_from_flat_position(position)
        grid = self.grid_levels[grid_idx]
        # Relative coordinates with respect to input:
        xmin, ymin, xmax, ymax = self.get_anchor_coords_wrt_its_input(position, make_absolute=False)
        # Relative coordinates with respect to original image:
        xmin = (xmin - grid.pad_rel) / (1 - grid.pad_rel)
        ymin = (ymin - grid.pad_rel) / (1 - grid.pad_rel)
        xmax = (xmax - grid.pad_rel) / (1 - grid.pad_rel)
        ymax = (ymax - grid.pad_rel) / (1 - grid.pad_rel)
        return xmin, ymin, xmax, ymax

    def decode_boxes_wrt_orig_tf(self, coords_enc):
        # coords_enc: (batch_size, nboxes, 4)

        # Coordinates wrt to anchor:
        coords_dec_wrt_anchor = CommonEncoding.decode_boxes_wrt_anchor_tf(coords_enc, self.opts)  # (batch_size, nboxes, 4) [xmin, ymin, width, height]

        # From coordinates wrt anchor to coordinates wrt to grid input:
        batch_size = coords_enc.shape[0]
        anchors_xmin_wrt_input = np.zeros(shape=(self.n_boxes), dtype=np.float32)
        anchors_ymin_wrt_input = np.zeros(shape=(self.n_boxes), dtype=np.float32)
        anchors_width_wrt_input = np.zeros(shape=(self.n_boxes), dtype=np.float32)
        anchors_height_wrt_input = np.zeros(shape=(self.n_boxes), dtype=np.float32)
        for pos in range(self.n_boxes):
            anc_xmin_wrt_input, anc_ymin_wrt_input, anc_xmax_wrt_input, anc_ymax_wrt_input = self.get_anchor_coords_wrt_its_input(pos)
            anchors_xmin_wrt_input[pos] = anc_xmin_wrt_input
            anchors_ymin_wrt_input[pos] = anc_ymin_wrt_input
            anchors_width_wrt_input[pos] = anc_xmax_wrt_input - anc_xmin_wrt_input
            anchors_height_wrt_input[pos] = anc_ymax_wrt_input - anc_ymin_wrt_input
        anchors_xmin_wrt_input = tf.constant(value=anchors_xmin_wrt_input, dtype=tf.float32)
        anchors_ymin_wrt_input = tf.constant(value=anchors_ymin_wrt_input, dtype=tf.float32)
        anchors_width_wrt_input = tf.constant(value=anchors_width_wrt_input, dtype=tf.float32)
        anchors_height_wrt_input = tf.constant(value=anchors_height_wrt_input, dtype=tf.float32)
        anchors_xmin_wrt_input_ext = tf.tile(tf.expand_dims(anchors_xmin_wrt_input, axis=0), [batch_size, 1])  # (batch_size, nboxes)
        anchors_ymin_wrt_input_ext = tf.tile(tf.expand_dims(anchors_ymin_wrt_input, axis=0), [batch_size, 1])  # (batch_size, nboxes)
        anchors_width_wrt_input_ext = tf.tile(tf.expand_dims(anchors_width_wrt_input, axis=0), [batch_size, 1])  # (batch_size, nboxes)
        anchors_height_wrt_input_ext = tf.tile(tf.expand_dims(anchors_height_wrt_input, axis=0), [batch_size, 1])  # (batch_size, nboxes)
        xmin_wrt_input = anchors_xmin_wrt_input_ext + coords_dec_wrt_anchor[..., 0] * anchors_width_wrt_input_ext  # (batch_size, nboxes)
        ymin_wrt_input = anchors_ymin_wrt_input_ext + coords_dec_wrt_anchor[..., 1] * anchors_height_wrt_input_ext  # (batch_size, nboxes)
        width_wrt_input = coords_dec_wrt_anchor[..., 2] * anchors_width_wrt_input_ext  # (batch_size, nboxes)
        height_wrt_input = coords_dec_wrt_anchor[..., 3] * anchors_height_wrt_input_ext  # (batch_size, nboxes)

        # From coordinates wrt input to coordinates wrt original image:
        pad_rel = np.zeros(shape=(self.n_boxes), dtype=np.float32)
        for pos in range(self.n_boxes):
            grid_idx = self.get_grid_idx_from_flat_position(pos)
            grid = self.grid_levels[grid_idx]
            pad_rel[pos] = grid.pad_rel
        pad_rel = tf.constant(value=pad_rel, dtype=tf.float32)
        pad_rel_ext = tf.tile(tf.expand_dims(pad_rel, axis=0), [batch_size, 1])  # (batch_size, nboxes)
        xmin_wrt_orig = (xmin_wrt_input - pad_rel_ext) / (1 - 2 * pad_rel_ext)  # (batch_size, nboxes)
        ymin_wrt_orig = (ymin_wrt_input - pad_rel_ext) / (1 - 2 * pad_rel_ext)  # (batch_size, nboxes)
        width_wrt_orig = width_wrt_input / (1 - 2 * pad_rel_ext)  # (batch_size, nboxes)
        height_wrt_orig = height_wrt_input / (1 - 2 * pad_rel_ext)  # (batch_size, nboxes)

        # Pack together:
        coords_dec = tf.stack([xmin_wrt_orig, ymin_wrt_orig, width_wrt_orig, height_wrt_orig], axis=-1)  # (batch_size, nboxes, 4) [xmin, ymin, width, height]

        return coords_dec  # (batch_size, nboxes, 4) [xmin, ymin, width, height]

    def make_loss_and_metrics(self, net_output, labels_enc):
        # net_output: (batch_size, nboxes, ?)
        # labels_enc: (batch_size, nboxes, nlabels)

        # Split net output:
        locs_enc = output_encoding.get_loc_enc(net_output)  # (batch_size, nboxes, 4)
        logits = output_encoding.get_logits(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes, nclasses)

        mask_match = gt_encoding.get_mask_match(labels_enc)  # (batch_size, nboxes)
        mask_neutral = gt_encoding.get_mask_neutral(labels_enc)  # (batch_size, nboxes)
        gt_class_ids = gt_encoding.get_class_id(labels_enc)  # (batch_size, nboxes)
        gt_coords = gt_encoding.get_coords_enc(labels_enc)  # (batch_size, nboxes)

        mask_match = tf.greater(mask_match, 0.5)  # (batch_size, nboxes)
        mask_neutral = tf.greater(mask_neutral, 0.5)  # (batch_size, nboxes)

        zeros = tf.zeros_like(mask_match, dtype=tf.float32)  # (batch_size, nboxes)

        conf_loss, accuracy_conf = classification_loss_and_metric(logits, mask_match, mask_neutral,
                                                                  gt_class_ids, zeros, self.opts.negative_ratio, self.n_boxes)
        tf.summary.scalar('losses/conf_loss', conf_loss)
        tf.summary.scalar('metrics/accuracy_conf', accuracy_conf)
        total_loss = conf_loss

        loc_loss, iou_mean = localization_loss_and_metric(locs_enc, mask_match, mask_neutral, gt_coords, zeros,
                                                          self.opts.loc_loss_factor, self.opts)
        tf.summary.scalar('losses/loc_loss', loc_loss)
        tf.summary.scalar('metrics/iou_mean', iou_mean)
        total_loss += loc_loss

        metrics = tf.stack([accuracy_conf, iou_mean])  # (2)

        # total_loss: ()
        return total_loss, metrics

    def encode_gt_batched(self, gt_boxes_batched):
        # gt_boxes_batched: List of length batch_size
        batch_size = len(gt_boxes_batched)
        labels_enc_batched = np.zeros(shape=(batch_size, self.n_boxes, self.n_labels), dtype=np.float32)
        for i in range(batch_size):
            labels_enc = self.encode_gt_from_array(gt_boxes_batched[i])  # (nboxes, nlabels)
            labels_enc_batched[i, ...] = labels_enc
        return labels_enc_batched


    def encode_gt_from_array(self, gt_boxes):
        # gt_boxes: (n_gt, 9) [class_id, xmin, ymin, width, height, pc, gt_idx, c_x_unclipped, c_y_unclipped]
        n_gt = gt_boxes.shape[0]
        if n_gt > 0:
            gt_coords = gt_boxes[:, 1:5]  # (n_gt, 4)
            gt_class_ids = gt_boxes[:, 0]  # (n_gt)
            gt_pc_incoming = gt_boxes[:, 5]  # (n_gt)
            gt_indices = gt_boxes[:, 6]  # (n_gt)

            pc, ar, dc = self.compute_pc_ar_dc(gt_coords)  # All variables have shape (n_gt, n_boxes)

            gt_pc_incoming_exp = np.expand_dims(gt_pc_incoming, axis=1)  # (n_gt, 1)
            gt_pc_incoming_exp = np.tile(gt_pc_incoming_exp, [1, self.n_boxes])  # (n_gt, n_boxes)
            pc = pc * gt_pc_incoming_exp  # (n_gt, n_boxes)

            # Positive boxes:
            mask_ar_low = ar > self.opts.threshold_ar_low  # (n_gt, n_boxes)
            mask_ar_high = ar < self.opts.threshold_ar_high  # (n_gt, n_boxes)
            mask_pc = pc > self.opts.threshold_pc  # (n_gt, n_boxes)
            mask_dc = dc < self.opts.threshold_dc  # (n_gt, n_boxes)
            mask_ar = mask_ar_low & mask_ar_high
            mask_thresholds = mask_ar & mask_pc & mask_dc  # (n_gt, n_boxes)
            mask_match = np.any(mask_thresholds, axis=0)  # (n_boxes)

            # Neutral boxes:
            mask_ar_low_neutral = ar > self.opts.threshold_ar_low_neutral  # (n_gt, n_boxes)
            mask_ar_high_neutral = ar < self.opts.threshold_ar_high_neutral  # (n_gt, n_boxes)
            mask_pc_neutral = pc > self.opts.threshold_pc_neutral  # (n_gt, n_boxes)
            mask_dc_neutral = dc < self.opts.threshold_dc_neutral  # (n_gt, n_boxes)
            mask_thresholds_neutral = mask_ar_low_neutral & mask_ar_high_neutral & mask_pc_neutral & mask_dc_neutral  # (n_gt, n_boxes)
            mask_neutral = np.any(mask_thresholds_neutral, axis=0)  # (n_boxes)
            mask_neutral = np.logical_and(mask_neutral, np.logical_not(mask_match))  # (n_boxes)

            dc_masked_match = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt, self.n_boxes), dtype=np.float32) ) # (n_gt, n_boxes)
            dc_masked_neutral = np.where(mask_thresholds_neutral, dc, np.infty * np.ones(shape=(n_gt, self.n_boxes), dtype=np.float32) ) # (n_gt, n_boxes)
            dc_masked = np.where(mask_match, dc_masked_match, dc_masked_neutral) # (n_gt, n_boxes)
            nearest_valid_box_idx = np.argmin(dc_masked, axis=0)  # (n_boxes)

            # Get the coordinates and the class id of the gt box matched:
            coordinates = np.take(gt_coords, nearest_valid_box_idx, axis=0)  # (n_boxes, 4)
            coordinates_enc = CommonEncoding.encode_boxes_wrt_anchor_np(coordinates, self.opts)  # (nboxes, 4)
            class_ids_pos = np.take(gt_class_ids, nearest_valid_box_idx, axis=0)  # (n_boxes)
            associated_gt_idx = np.take(gt_indices, nearest_valid_box_idx)  # (n_boxes)

            # Negative boxes:
            mask_negative = np.logical_and(np.logical_not(mask_match), np.logical_not(mask_neutral))  # (n_boxes)
            mask_negative_x_4 = np.tile(np.expand_dims(mask_negative, axis=-1), [1, 4])  # (n_boxes, 4)
            background_ids = np.ones(shape=(self.n_boxes), dtype=np.int32) * self.background_id
            background_coords = np.zeros(shape=(self.n_boxes, 4), dtype=np.float32)  # (nboxes, 4)
            class_ids = np.where(mask_negative, background_ids, class_ids_pos)  # (n_boxes)
            coordinates_enc = np.where(mask_negative_x_4, background_coords, coordinates_enc)  # (n_boxes, 4)
            associated_gt_idx = np.where(mask_negative, -1 * np.ones(shape=(self.n_boxes), dtype=np.float32), associated_gt_idx)  # (n_boxes)

            # Put all together in one array:
            labels_enc = np.stack([mask_match.astype(np.float32),
                                   mask_neutral.astype(np.float32),
                                   class_ids,
                                   associated_gt_idx], axis=-1)
            labels_enc = np.concatenate([coordinates_enc, labels_enc], axis=-1)  # (n_boxes, n_labels)

        else:
            labels_enc = np.zeros(shape=(self.n_boxes, self.n_labels), dtype=np.float32)
            labels_enc[:, 6] = self.background_id

        return labels_enc  # (n_boxes, n_labels)

    # Compute the 'Percent Contained', the 'Area Ratio' and the 'Distance to Center' between ground truth boxes and and anchor boxes.
    def compute_pc_ar_dc(self, gt_coords):
        # gt_boxes (n_gt, 4) [xmin, ymin, width, height]  # Parameterized with the top-left coordinates, and the width and height.
        # self.anchors_coordinates is parameterized with the corner coordinates (xmin, ymin, xmax, ymax)
        # Coordinates are relative (between 0 and 1)
        # We consider the square containing the ground truth box.

        n_gt = gt_coords.shape[0]

        orig_boxes = np.stack([gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 0] + gt_coords[:, 2], gt_coords[:, 1] + gt_coords[:, 3]], axis=1)
        orig_boxes_expanded = np.expand_dims(orig_boxes, axis=1)  # (n_gt, 1, 4)
        orig_boxes_expanded = np.tile(orig_boxes_expanded, (1, self.n_boxes, 1))  # (n_gt, nboxes, 4) [xmin, ymin, xmax, ymax]

        gt_boxes_xcenter = gt_coords[:, 0] + gt_coords[:, 2] / 2.0
        gt_boxes_ycenter = gt_coords[:, 1] + gt_coords[:, 3] / 2.0
        gt_boxes_maxside = np.maximum(gt_coords[:, 2], gt_coords[:, 3])
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

        return pc, ar, dc # (n_gt, nboxes)

    def write_anchor_info_file(self, info_file_path, softmax, pred_class, cm_pred, img_idx, anchor_pos, coords_enc, coords_dec):
        # softmax: (nclasses)
        with open(info_file_path, 'w') as fid:
            fid.write('softmax =')
            for j in range(self.nclasses):
                if j != 0:
                    fid.write(',')
                fid.write(' ' + str(softmax[img_idx, anchor_pos, j]))
            fid.write('\n')
            fid.write('predicted_class = ' + str(pred_class) + ' (' + self.classnames[pred_class] + ')\n')
            fid.write('conf = ' + str(softmax[img_idx, anchor_pos, pred_class]) + '\n')
            grid_idx = self.get_grid_idx_from_flat_position(anchor_pos)
            grid = self.grid_levels[grid_idx]
            fid.write('grid_idx = ' + str(grid_idx) + '\n')
            xmin_a_i, ymin_a_i, xmax_a_i, ymax_a_i = self.get_anchor_coords_wrt_its_input(anchor_pos)
            fid.write('anchor coords input [xmin, ymin, xmax, ymax] = ' + str([xmin_a_i, ymin_a_i, xmax_a_i, ymax_a_i]) + '\n')
            fid.write('anchor coords orig [xmin, ymin, width, height] = ' + str([self.anchors_coordinates[anchor_pos, 0],
                                                                            self.anchors_coordinates[anchor_pos, 1],
                                                                            self.anchors_w[anchor_pos],
                                                                            self.anchors_h[anchor_pos]]) + '\n')
            fid.write('coords_enc = ' + str(coords_enc) + '\n')
            fid.write('coords_dec = ' + str(coords_dec) + '\n')
            x0_r_a = coords_dec[0]
            x0_r_p = xmin_a_i + x0_r_a * (xmax_a_i - xmin_a_i)
            x0_r_o = min(max((x0_r_p - grid.pad_rel) / (1 - 2 * grid.pad_rel), 0), 1)
            fid.write('x0_r_a = ' + str(x0_r_a) + '\n')
            fid.write('x0_r_p = ' + str(x0_r_p) + '\n')
            fid.write('x0_r_o = ' + str(x0_r_o) + '\n')
            y0_r_a = coords_dec[1]
            y0_r_p = ymin_a_i + y0_r_a * (ymax_a_i - ymin_a_i)
            y0_r_o = min(max((y0_r_p - grid.pad_rel) / (1 - grid.pad_rel), 0), 1)
            fid.write('y0_r_a = ' + str(y0_r_a) + '\n')
            fid.write('y0_r_p = ' + str(y0_r_p) + '\n')
            fid.write('y0_r_o = ' + str(y0_r_o) + '\n')
            if self.opts.predict_cm:
                fid.write('cm_pred = ' + str(cm_pred[img_idx, anchor_pos]) + '\n')

    def write_debug_info(self, *arg):
        # arg: [net_output, inputs0, ..., inputsN]
        # net_output: (batch_size, nboxes, 4+nclasses)
        # inputsX: (batch_size, height, width, 3)
        net_output = arg[0]  # (batch_size, nboxes, n_channels_last)
        inputs_all_sizes = []
        for i in range(1, len(arg)):
            inputs_all_sizes.append(arg[i])
        batch_size = net_output.shape[0]

        localizations_enc = output_encoding.get_loc_enc(net_output)  # (batch_size, nboxes, 4)
        logits = output_encoding.get_logits(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes, nclasses)
        softmax = tools.softmax_np(logits)  # (batch_size, nboxes, nclasses)
        cm_pred_enc = output_encoding.get_cm_enc(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes) or None
        cm_pred = CommonEncoding.decode_cm_np(cm_pred_enc)  # (batch_size, nboxes) or None

        self.batch_count_debug += 1
        batch_dir = os.path.join(self.debug_dir, 'batch' + str(self.batch_count_debug))
        os.makedirs(batch_dir)
        dir_images = []
        for img_idx in range(batch_size):
            dir_this_image = os.path.join(batch_dir, 'img' + str(img_idx))
            dir_images.append(dir_this_image)
            os.makedirs(dir_this_image)
        for pos in range(self.n_boxes):
            grid_idx = self.get_grid_idx_from_flat_position(pos)
            inputs_this_box = inputs_all_sizes[grid_idx]  # (batch_size, height, width, 3)
            anchor_coords = self.get_anchor_coords_wrt_its_input(pos, True)  # [xmin, ymin, xmax, ymax]
            anc_xmin = anchor_coords[0]
            anc_ymin = anchor_coords[1]
            anc_xmax = anchor_coords[2]
            anc_ymax = anchor_coords[3]
            for img_idx in range(batch_size):
                dir_img = dir_images[img_idx]
                img = inputs_this_box[img_idx, anc_ymin:anc_ymax, anc_xmin:anc_xmax].copy()  # (receptive_field_size, receptive_field_size, 3)
                img = tools.add_mean_again(img)
                path_to_save = os.path.join(dir_img, 'pos' + str(pos) + '.png')
                coords_enc = localizations_enc[img_idx, pos, :]  # (4)
                coords_dec = CommonEncoding.decode_boxes_wrt_anchor_np(coords_enc, self.opts)  # (4)
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
                # Save info file:
                info_file_path = os.path.join(dir_img, 'pos' + str(pos) + '_info.txt')
                self.write_anchor_info_file(info_file_path, softmax, predicted_class, cm_pred, img_idx, pos, coords_enc, coords_dec)
        return net_output


def classification_loss_and_metric(pred_conf, mask_match, mask_neutral, gt_class, zeros, negative_ratio, n_boxes):
    # pred_conf: (batch_size, nboxes, nclasses)
    # mask_match: (batch_size, nboxes)
    # mask_negative: (batch_size, nboxes)
    # gt_class: (batch_size, nboxes)
    # zeros: (batch_size, nboxes)
    with tf.variable_scope('conf_loss'):
        gt_class_int = tf.cast(gt_class, tf.int32)
        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_int, logits=pred_conf, name='loss_orig')  # (batch_size, nboxes)
        loss_positives = tf.where(mask_match, loss_orig, zeros, name='loss_positives')  # (batch_size)
        n_positives = tf.reduce_sum(tf.cast(mask_match, tf.int32), name='n_positives')  # ()
        loss_pos_scaled = tf.divide(loss_positives, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_pos_scaled')  # (batch_size, nboxes)

        # Hard negative mining:
        mask_negatives = tf.logical_and(tf.logical_not(mask_match), tf.logical_not(mask_neutral), name='mask_negatives')  # (batch_size, nboxes)
        loss_negatives = tf.where(mask_negatives, loss_orig, zeros, name='loss_negatives')
        n_negatives = tf.reduce_sum(tf.cast(mask_negatives, tf.int32), name='n_negatives')  # ()
        n_negatives_keep = tf.cast(tf.minimum(tf.maximum(1, n_positives * negative_ratio), n_negatives), tf.int32)
        # n_positives = tf.Print(n_positives, [n_positives], 'n_positives')
        # n_negatives_keep = tf.Print(n_negatives_keep, [n_negatives_keep], 'n_negatives_keep')
        loss_negatives_flat = tf.reshape(loss_negatives, [-1])  # (batch_size * nboxes)
        _, indices = tf.nn.top_k(loss_negatives_flat, k=n_negatives_keep, sorted=False)  # (?)
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(loss_negatives_flat))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [-1, n_boxes]), tf.bool)  # (batch_size, nboxes)
        loss_negatives = tf.where(negatives_keep, loss_orig, zeros, name='loss_negatives')
        loss_neg_scaled = tf.divide(loss_negatives, tf.maximum(tf.cast(n_negatives, tf.float32), 1), name='loss_neg_scaled')  # (batch_size, nboxes)

        # Join positives and negatives loss:
        loss_conf = tf.reduce_sum(loss_pos_scaled + loss_neg_scaled, name='loss_conf')  # ()

        # Metric:
        predicted_class = tf.argmax(pred_conf, axis=-1, output_type=tf.int32)  # (batch_size, nboxes)
        hits = tf.cast(tf.equal(gt_class_int, predicted_class), tf.float32)  # (batch_ize, nboxes)
        hits_no_neutral = tf.where(tf.logical_not(mask_neutral), hits, zeros)  # (batc_size, nboxes)
        n_hits = tf.reduce_sum(hits_no_neutral)  # ()
        accuracy_conf = tf.divide(n_hits, tf.maximum(tf.cast(n_negatives + n_positives, tf.float32), 1))  # ()

    return loss_conf, accuracy_conf


def localization_loss_and_metric(pred_coords, mask_match, mask_neutral, gt_coords, zeros, loc_loss_factor, opts):
    # pred_coords: (batch_size, nboxes, 4)  encoded
    # mask_match: (batch_size, nboxes)
    # mask_neutral: (batch_size, nboxes)
    # gt_coords: (batch_size, nboxes, 4)  encoded
    # zeros: (batch_size, nboxes)
    with tf.variable_scope('loc_loss'):
        localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, pred_coords)  # (batch_size, nboxes)
        valids_for_loc = tf.logical_or(mask_match, mask_neutral)  # (batch_size, nboxes)
        n_valids = tf.reduce_sum(tf.cast(valids_for_loc, tf.int32), name='n_valids')  # ()
        n_valids_safe = tf.maximum(tf.cast(n_valids, tf.float32), 1)
        localization_loss_matches = tf.where(valids_for_loc, localization_loss, zeros, name='loss_match')  # (batch_size, nboxes)
        loss_loc_summed = tf.reduce_sum(localization_loss_matches, name='loss_summed')  # ()
        loss_loc_scaled = tf.divide(loss_loc_summed, n_valids_safe, name='loss_scaled')  # ()
        loss_loc = tf.multiply(loss_loc_scaled, loc_loss_factor, name='loss_loc')  # ()

        # Metric:
        pred_coords_dec = CommonEncoding.decode_boxes_wrt_anchor_tf(pred_coords, opts)  # (batch_size, nboxes, 4)
        gt_coords_dec = CommonEncoding.decode_boxes_wrt_anchor_tf(gt_coords, opts)  # (batch_size, nboxes, 4)
        iou = compute_iou_tf(pred_coords_dec, gt_coords_dec)  # (batch_size, nboxes)
        iou_matches = tf.where(valids_for_loc, iou, zeros)  # (batch_size, nboxes)
        iou_summed = tf.reduce_sum(iou_matches)  # ()
        iou_mean = tf.divide(iou_summed, n_valids_safe, name='iou_mean')  # ()
    return loss_loc, iou_mean


def compute_iou_np(boxes1, boxes2):
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


def compute_iou_tf(boxes1, boxes2):
    # boxes1: (..., 4) [xmin, ymin, width, height]
    # boxes2: (..., 4) [xmin, ymin, width, height]
    xmin = tf.maximum(boxes1[..., 0], boxes2[..., 0])  # (batch_size)
    ymin = tf.maximum(boxes1[..., 1], boxes2[..., 1])  # (batch_size)
    xmax = tf.minimum(boxes1[..., 0] + boxes1[..., 2], boxes2[..., 0] + boxes2[..., 2])  # (batch_size)
    ymax = tf.minimum(boxes1[..., 1] + boxes1[..., 3], boxes2[..., 1] + boxes2[..., 3])  # (batch_size)
    intersection_area = tf.maximum((xmax - xmin), 0.0) * tf.maximum((ymax - ymin), 0.0)  # (batch_size)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # (batch_size)
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # (batch_size)
    union_area = boxes1_area + boxes2_area - intersection_area  # (batch_size)
    iou = intersection_area / union_area  # (batch_size)
    return iou




