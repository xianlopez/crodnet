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
        self.max_pad_rel = -1
        self.max_pad_abs = -1
        self.input_image_size_w_pad = -1
        self.expected_n_boxes = self.get_expected_num_boxes()
        self.n_comparisons = -1
        self.metric_names = ['mAP']
        self.n_metrics = len(self.metric_names)
        self.make_comparison_op()
        self.batch_count_debug = 0
        self.th_conf = th_conf
        self.classnames = classnames
        if self.opts.debug:
            self.debug_dir = os.path.join(outdir, 'debug')
            os.makedirs(self.debug_dir)

    def make_comparison_op(self):
        self.CR1 = tf.placeholder(shape=(self.opts.lcr), dtype=tf.float32)
        self.CR2 = tf.placeholder(shape=(self.opts.lcr), dtype=tf.float32)
        CRs = tf.stack([self.CR1, self.CR2], axis=0)  # (2, lcr)
        CRs = tf.expand_dims(CRs, axis=0)  # (1, 2, lcr)
        self.comparison_op = network.comparison(CRs, self.opts.lcr)  # (1, 2)
        self.comparison_op = tf.squeeze(self.comparison_op)  # (2)
        softmax = tf.nn.softmax(self.comparison_op, axis=-1)  # (2)
        self.pseudo_distance = softmax[0]  # ()

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

    def assign_comparisons_between_anchors(self):
        total_number_of_comparisons = 0
        self.comparisons_references = []
        for i in range(self.n_boxes):
            anchors_to_compare = []
            for j in range(i+1, self.n_boxes):
                box1 = np.array([self.anchors_coordinates[i, 0],
                                 self.anchors_coordinates[i, 1],
                                 self.anchors_w[i],
                                 self.anchors_h[i]])  # (xmin, ymin, width, height)
                box2 = np.array([self.anchors_coordinates[j, 0],
                                 self.anchors_coordinates[j, 1],
                                 self.anchors_w[j],
                                 self.anchors_h[j]])  # (xmin, ymin, width, height)
                iou = compute_iou_multi_dim(box1, box2)
                if iou >= self.opts.min_iou_to_compare:
                    anchors_to_compare.append(j)
            n_comp_this_anchor = len(anchors_to_compare)
            total_number_of_comparisons += n_comp_this_anchor
            # print('Anchor ' + str(i) + ': ' + str(n_comp_this_anchor) + ' comparisons.')
            self.comparisons_references.append(anchors_to_compare)
        mean_comparisons = total_number_of_comparisons / float(self.n_boxes)
        print('Mean number of comparisons per anchor: ' + str(mean_comparisons))

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
        self.n_boxes = 0
        all_outputs = []
        all_crs = []
        for i in range(len(self.grid_levels)):
            grid = self.grid_levels[i]
            print('')
            print('Defining network for input size ' + str(grid.input_size) + ' (pad: ' + str(grid.pad_abs) + ')')
            common_representation = network.common_representation(inputs_all_sizes[i], self.opts.lcr)
            net = network.prediction_path(common_representation, self.opts, self.nclasses, self.opts.predict_pc, self.opts.predict_dc)
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
            if (grid.input_size_w_pad - network.receptive_field_size) / network.step_in_pixels + 1 != grid.output_shape:
                raise Exception('Inconsistent step for input size ' + str(grid.input_size_w_pad) + '. Grid size is ' + str(grid.output_shape) + '.')
        assert self.n_boxes == self.get_expected_num_boxes(), 'Expected number of boxes differs from the real number.'
        all_outputs = tf.concat(all_outputs, axis=1, name='all_outputs')  # (batch_size, nboxes, 4+nclasses)
        all_crs = tf.concat(all_crs, axis=1, name='all_crs')  # (batch_size, nboxes, lcr)
        self.input_image_size_w_pad = int(np.ceil(float(self.input_image_size) / (1 - 2 * self.max_pad_rel)))
        self.max_pad_abs = int(np.round(self.max_pad_rel * self.input_image_size_w_pad))
        self.input_image_size_w_pad = self.input_image_size + 2 * self.max_pad_abs
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
        self.assign_comparisons_between_anchors()
        if self.opts.debug:
            debug_inputs = [net_output]
            debug_inputs.extend(inputs_all_sizes)
            net_output_shape = net_output.shape
            net_output = tf.py_func(self.write_debug_info, debug_inputs, (tf.float32))
            net_output.set_shape(net_output_shape)
        localizations, softmax, pc, dc = self.obtain_localizations_and_softmax(net_output)
        return localizations, softmax, CRs, pc, dc

    def obtain_localizations_and_softmax(self, net_output):
        # net_output: (batch_size, nboxes, ?)

        # Split net output:
        locs_enc = output_encoding.get_loc_enc(net_output)  # (batch_size, nboxes, 4)
        logits = output_encoding.get_logits(net_output, self.opts.predict_pc, self.opts.predict_dc)  # (batch_size, nboxes, nclasses)
        pc = output_encoding.get_pc(net_output, self.opts.predict_pc, self.opts.predict_dc)  # (batch_size, nboxes) or None
        dc = output_encoding.get_dc(net_output, self.opts.predict_pc, self.opts.predict_dc)  # (batch_size, nboxes) or None

        # Decode
        localizations_dec = self.decode_boxes_wrt_padded_tf(locs_enc)  # (batch_size, nboxes, 4)
        softmax = tf.nn.softmax(logits, axis=-1)  # (batch_size, nboxes, nclasses)

        return localizations_dec, softmax, pc, dc


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

    def decode_boxes_wrt_padded_tf(self, coords_enc):
        # coords_enc: (..., 4)
        dcx_enc = coords_enc[..., 0]
        dcy_enc = coords_enc[..., 1]
        w_enc = coords_enc[..., 2]
        h_enc = coords_enc[..., 3]

        # Decoding step:
        dcx_rel, dcy_rel, w_rel, h_rel = CommonEncoding.decoding_split_tf(dcx_enc, dcy_enc, w_enc, h_enc, self.opts)

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




