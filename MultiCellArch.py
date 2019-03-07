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
        localizations, softmax, pc, dc, cm = self.obtain_localizations_and_softmax(net_output)
        return localizations, softmax, CRs, pc, dc, cm

    def obtain_localizations_and_softmax(self, net_output):
        # net_output: (batch_size, nboxes, ?)

        # Split net output:
        locs_enc = output_encoding.get_loc_enc(net_output)  # (batch_size, nboxes, 4)
        logits = output_encoding.get_logits(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes, nclasses)
        pc_enc = output_encoding.get_pc_enc(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes) or None
        dc_enc = output_encoding.get_dc_enc(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes) or None
        cm_enc = output_encoding.get_cm_enc(net_output, self.opts.predict_pc, self.opts.predict_dc, self.opts.predict_cm)  # (batch_size, nboxes) or None

        # Decode
        localizations_dec = self.decode_boxes_wrt_orig_tf(locs_enc)  # (batch_size, nboxes, 4)
        softmax = tf.nn.softmax(logits, axis=-1)  # (batch_size, nboxes, nclasses)
        if self.opts.predict_pc:
            pc = CommonEncoding.decode_pc_or_dc_tf(pc_enc)
        else:
            pc = None
        if self.opts.predict_dc:
            dc = CommonEncoding.decode_pc_or_dc_tf(dc_enc)
        else:
            dc = None
        if self.opts.predict_cm:
            cm = CommonEncoding.decode_cm_tf(cm_enc)
        else:
            cm = None

        return localizations_dec, softmax, pc, dc, cm


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




