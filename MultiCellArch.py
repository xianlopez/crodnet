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
    def __init__(self, options, nclasses, is_training, outdir):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.n_grids = len(self.opts.grid_levels_size_pad)
        self.grid_levels = []
        for i in range(self.n_grids):
            size_pad = self.opts.grid_levels_size_pad[i]
            self.grid_levels.append(GridLevel(size_pad[0], size_pad[1]))
        self.grid_levels.sort(key=operator.attrgetter('input_size'))
        self.is_training = is_training
        self.n_boxes = -1
        self.receptive_field_size = 352
        self.max_pad_rel = -1
        self.max_pad_abs = -1
        self.input_image_size_w_pad = -1
        self.expected_n_boxes = self.get_expected_num_boxes()
        self.encoded_gt_shape = (self.expected_n_boxes, 9)
        self.outdir = outdir
        self.n_comparisons = -1

        self.input_crops_dir = os.path.join(self.outdir, 'input_crops')

        if os.name == 'nt':
            _, self.classnames = tools.process_dataset_config(r'D:\datasets\VOC0712_filtered\dataset_info.xml')
        elif os.name == 'posix':
            _, self.classnames = tools.process_dataset_config('/home/xian/datasets/VOC0712_filtered/dataset_info.xml')
        else:
            raise Exception('Unexpected OS')

        self.classnames.append('background')

    def get_expected_num_boxes(self):
        expected_n_boxes = 0
        for grid in self.grid_levels:
            dim_this_grid = 1 + (grid.input_size_w_pad - self.receptive_field_size) // self.opts.step_in_pixels
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

    def add_mean_again(self, image):
        mean = [123.0, 117.0, 104.0]
        mean = np.reshape(mean, [1, 1, 3])
        image = image + mean
        return image

    def get_input_crop(self, batch_idx, position, inputs_all_sizes):
        grid_idx, _, _ = self.get_grid_coordinates_from_flat_position(position)
        xmin, ymin, xmax, ymax = self.get_anchor_coords_wrt_its_input(position, make_absolute=True)
        # Get the input image that corresponds:
        inputs = inputs_all_sizes[grid_idx]
        image = inputs[batch_idx, :, :, :].copy()
        image = self.add_mean_again(image)
        # Crop it:
        crop = image[ymin:ymax, xmin:xmax]
        return crop

    def write_predictions(self, inputs0, labels_enc, predictions, filenames, net_output, labels2):
        # labels_enc (batch_size, nboxes x 9)
        # predictions (batch_size, nboxes, nclasses)
        # print('')
        # print('write_predictions')

        inputs_all_sizes = [inputs0]

        correct_dir = os.path.join(self.outdir, 'correct')
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)
        wrong_dir = os.path.join(self.outdir, 'wrong')
        if not os.path.exists(wrong_dir):
            os.makedirs(wrong_dir)
        neutral_dir = os.path.join(self.outdir, 'neutral')
        if not os.path.exists(neutral_dir):
            os.makedirs(neutral_dir)

        batch_size = labels_enc.shape[0]
        for batch_idx in range(batch_size):
            # print('')
            # print('batch idx ' + str(batch_idx))
            name = filenames[batch_idx].decode(sys.getdefaultencoding())
            mask_match = labels_enc[batch_idx, :, 4]
            mask_neutral = labels_enc[batch_idx, :, 5]
            mask_match = mask_match > 0.5
            mask_neutral = mask_neutral > 0.5
            mask_negative = np.logical_and(np.logical_not(mask_match), np.logical_not(mask_neutral))
            box_coords_enc = labels_enc[batch_idx, :, :4]  # (nboxes, 4)
            box_coords_raw = self.decode_boxes(box_coords_enc)  # (nboxes, 4) # Padded image reference [xmin, ymin, width, height]
            class_ids = labels_enc[batch_idx, :, 6].astype(np.int32)
            class_id = class_ids[0] # We only consider one box.
            predicted_class_id = np.argmax(predictions[batch_idx, 0, :])
            percent_contained = labels_enc[batch_idx, :, 8]  # (nboxes)
            crop = self.get_input_crop(batch_idx, 0, inputs_all_sizes)
            if mask_match[0]:
                # Box coordinates with repect to the crop (the anchor):
                box_coords_wrt_anchor_abs = self.box_coords_wrt_padded_2_box_coords_wrt_anchor(box_coords_raw[0, :], 0, True)
                # Draw the object:
                cv2.rectangle(crop, (box_coords_wrt_anchor_abs[0], box_coords_wrt_anchor_abs[1]),
                              (box_coords_wrt_anchor_abs[0] + box_coords_wrt_anchor_abs[2],
                               box_coords_wrt_anchor_abs[1] + box_coords_wrt_anchor_abs[3]), (0, 0, 255), 2)
                # Get the predicted and ground truth class names:
                predicted_class = self.classnames[predicted_class_id]
                gt_class = self.classnames[class_id]
                # Check if the prediction is right or wrong:
                if class_id == predicted_class_id:
                    image_folder = os.path.join(correct_dir, gt_class)
                else:
                    image_folder = os.path.join(wrong_dir, gt_class)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                # Write crop:
                pc = int(np.round(percent_contained[0] * 100))
                if pc < 100:
                    print('POSITIVE WITH PC ' + str(pc))
                image_path = os.path.join(image_folder, name + '_predicted_' + predicted_class + '_pc_' + str(pc) + '.png')
            elif mask_neutral[0]:
                image_path = os.path.join(neutral_dir, name + '.png')
            elif mask_negative[0]:
                # Get the predicted and ground truth class names:
                predicted_class = self.classnames[predicted_class_id]
                assert class_id == self.background_id, 'Negative box does not have background class id'
                gt_class = 'background'
                # Check if the prediction is right or wrong:
                if class_id == predicted_class_id:
                    image_folder = os.path.join(correct_dir, gt_class)
                else:
                    image_folder = os.path.join(wrong_dir, gt_class)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                # Write crop:
                image_path = os.path.join(image_folder, name + '_predicted_' + predicted_class + '.png')
            else:
                raise Exception('Box is not positive, not neutral and not negative.')
            cv2.imwrite(image_path, cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR))
        return labels2

    def write_input_crops_1grid(self, inputs0, labels, filenames, all_outputs):
        inputs_all_sizes = [inputs0]
        labels = self.write_input_crops(inputs_all_sizes, labels, filenames, all_outputs)
        return labels

    def write_input_crops_2grids(self, inputs0, inputs1, labels, filenames, all_outputs):
        inputs_all_sizes = [inputs0, inputs1]
        labels = self.write_input_crops(inputs_all_sizes, labels, filenames, all_outputs)
        return labels

    def write_input_crops_3grids(self, inputs0, inputs1, inputs2, labels, filenames, all_outputs):
        inputs_all_sizes = [inputs0, inputs1, inputs2]
        labels = self.write_input_crops(inputs_all_sizes, labels, filenames, all_outputs)
        return labels

    def write_net_output(self, softmax, filename, directory):
        # softmax: (nboxes, nclasses)
        file_path = os.path.join(directory, filename + '_net_output.csv')
        print('net_output file path: ' + file_path)
        with open(file_path, 'w') as fid:
            for i in range(self.n_boxes):
                fid.write('\n')
                fid.write('Box ' + str(i) + '\n')
                fid.write('softmax: ;')
                for cl_idx in range(self.nclasses):
                    fid.write(str(softmax[i, cl_idx]) + ';')
                fid.write('\n')
                winning_class = np.argmax(softmax[i, :])
                classname = self.classnames[winning_class]
                max_conf = softmax[i, winning_class]
                fid.write('Winning class: ;' + str(winning_class) + ';' + classname + '\n')
                fid.write('Max conf: ;' + str(max_conf) + '\n')

    def write_input_crops(self, inputs_all_sizes, labels, filenames, all_outputs):
        # inputs_all_sizes: list of elements of shape (batch_size, this_input_size, this_input_size, 3)
        # labels: (batch_size, nboxes, 9)
        # filenames: (batch_size)
        # all_outputs: (batch_size, nboxes, ?)
        # print('write_input_crops')

        net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(all_outputs, self.opts, self.nclasses)
        # net_output_conf: (batch_size, nboxes, nclasses)
        # net_output_coords: (batch_size, nboxes, ?)
        # net_output_pc: (batch_size, nboxes)
        if self.opts.box_per_class:
            # net_output_coords: (batch_size, nboxes, 4 * nclasses)
            selected_coords = CommonEncoding.get_selected_coords_np(net_output_conf, net_output_coords, self.nclasses, self.opts.box_per_class)  # (batch_size, nboxes, 4)
            pred_coords_raw = self.decode_boxes(selected_coords)  # (batch_size, nboxes, 4) [xmin, ymin, width, height]
        else:
            # net_output_coords: (batch_size, nboxes, 4)
            pred_coords_raw = self.decode_boxes(net_output_coords)  # (batch_size, nboxes, 4) [xmin, ymin, width, height]

        e_x = np.exp(net_output_conf)  # (batch_size, nboxes, nclasses)
        softmax = e_x / np.tile(np.sum(e_x, axis=-1, keepdims=True), [1, 1, self.nclasses])  # (batch_size, nboxes)

        batch_size = labels.shape[0]
        for batch_idx in range(batch_size):
            name = filenames[batch_idx].decode(sys.getdefaultencoding())
            # print('image ' + name)
            mask_match = labels[batch_idx, :, 4]
            mask_neutral = labels[batch_idx, :, 5]
            # print('mask_match')
            # print(mask_match)
            # print('mask_neutral')
            # print(mask_neutral)
            mask_match = mask_match > 0.5
            mask_neutral = mask_neutral > 0.5
            mask_negative = np.logical_and(np.logical_not(mask_match), np.logical_not(mask_neutral))
            class_ids = labels[batch_idx, :, 6]
            box_coords_enc = labels[batch_idx, :, :4]  # (nboxes, 4)
            box_coords_raw = self.decode_boxes(box_coords_enc)  # (nboxes, 4) # Padded image reference [xmin, ymin, width, height]
            # print('n_positives = ' + str(np.sum(mask_match.astype(np.int32))))
            # print('n_neutrals = ' + str(np.sum(mask_neutral.astype(np.int32))))
            # print('n_negatives = ' + str(np.sum(mask_negative.astype(np.int32))))
            # print('box_coords_raw = ' + str(box_coords_raw))
            image_dir = os.path.join(self.input_crops_dir, name)
            positives_dir = os.path.join(image_dir, 'positives')
            negatives_dir = os.path.join(image_dir, 'negatives')
            neutrals_dir = os.path.join(image_dir, 'neutrals')
            if not os.path.exists(self.input_crops_dir):
                os.makedirs(self.input_crops_dir)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            if not os.path.exists(positives_dir):
                os.makedirs(positives_dir)
            if not os.path.exists(negatives_dir):
                os.makedirs(negatives_dir)
            if not os.path.exists(neutrals_dir):
                os.makedirs(neutrals_dir)
            for i in range(len(inputs_all_sizes)):
                image_grid = inputs_all_sizes[i][batch_idx, :, :, :].copy()
                image_grid = self.add_mean_again(image_grid)
                cv2.imwrite(os.path.join(image_dir, 'image_grid' + str(i) + '.png'), cv2.cvtColor(image_grid.astype(np.uint8), cv2.COLOR_RGB2BGR))
            for position in range(self.n_boxes):
                # print('position ' + str(position))
                self.write_net_output(softmax[batch_idx, :, :], name, image_dir)
                crop = self.get_input_crop(batch_idx, position, inputs_all_sizes)
                # Save it:
                if mask_negative[position]:  # negative
                    image_path = os.path.join(negatives_dir, 'pos_' + str(position) + '_background.png')
                elif mask_match[position]:  # positive
                    id_class = int(np.round(class_ids[position]))
                    image_path = os.path.join(positives_dir, 'pos_' + str(position) + '_class_' + self.classnames[id_class] + '.png')
                    # Box coordinates with repect to the crop (the anchor):
                    box_coords_wrt_anchor_abs = self.box_coords_wrt_padded_2_box_coords_wrt_anchor(box_coords_raw[position, :], position, True)
                    # print('box_coords_wrt_anchor_abs = ' + str(box_coords_wrt_anchor_abs))
                    # Draw the object:
                    cv2.rectangle(crop, (box_coords_wrt_anchor_abs[0], box_coords_wrt_anchor_abs[1]),
                                  (box_coords_wrt_anchor_abs[0] + box_coords_wrt_anchor_abs[2],
                                   box_coords_wrt_anchor_abs[1] + box_coords_wrt_anchor_abs[3]), (0, 0, 255), 2)
                else:  # neutral
                    image_path = os.path.join(neutrals_dir, 'pos_' + str(position) + '_neutral.png')
                # Prediction coordinates:
                print()
                pred_coords_wrt_anchor_abs = self.box_coords_wrt_padded_2_box_coords_wrt_anchor(pred_coords_raw[batch_idx, position, :], position, True)
                cv2.rectangle(crop, (pred_coords_wrt_anchor_abs[0], pred_coords_wrt_anchor_abs[1]),
                              (pred_coords_wrt_anchor_abs[0] + pred_coords_wrt_anchor_abs[2],
                               pred_coords_wrt_anchor_abs[1] + pred_coords_wrt_anchor_abs[3]), (255, 165, 0), 2)
                image_path = tools.ensure_new_path(image_path)
                # print('writing for position ' + str(position))
                cv2.imwrite(image_path, cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR))
        return labels

    def box_coords_wrt_padded_2_box_coords_wrt_anchor(self, box_coords_wrt_pad, position, convert_to_abs=False):
        box_xmin_wrt_pad = box_coords_wrt_pad[0]
        box_ymin_wrt_pad = box_coords_wrt_pad[1]
        box_width_wrt_pad = box_coords_wrt_pad[2]
        box_height_wrt_pad = box_coords_wrt_pad[3]
        anchor_xmin_wrt_pad, anchor_ymin_wrt_pad, anchor_xmax_wrt_pad, anchor_ymax_wrt_pad = self.get_anchor_coords_wrt_padded(position)
        # print('anchor_coords_wrt_padded: ' + str([anchor_xmin_wrt_pad, anchor_ymin_wrt_pad, anchor_xmax_wrt_pad, anchor_ymax_wrt_pad]))
        anchor_width_wrt_pad = anchor_xmax_wrt_pad - anchor_xmin_wrt_pad
        anchor_height_wrt_pad = anchor_ymax_wrt_pad - anchor_ymin_wrt_pad
        # print('anchor_width_wrt_pad = ' + str(anchor_width_wrt_pad))
        # print('anchor_height_wrt_pad = ' + str(anchor_height_wrt_pad))
        box_xmin_wrt_anchor = (box_xmin_wrt_pad - anchor_xmin_wrt_pad) / anchor_width_wrt_pad
        box_ymin_wrt_anchor = (box_ymin_wrt_pad - anchor_ymin_wrt_pad) / anchor_height_wrt_pad
        box_width_wrt_anchor = box_width_wrt_pad / anchor_width_wrt_pad
        box_height_wrt_anchor = box_height_wrt_pad / anchor_height_wrt_pad
        box_coords_wrt_anchor = np.array([box_xmin_wrt_anchor, box_ymin_wrt_anchor, box_width_wrt_anchor, box_height_wrt_anchor], dtype=np.float32)
        if convert_to_abs:
            box_coords_wrt_anchor = np.clip(np.round(box_coords_wrt_anchor * self.receptive_field_size).astype(np.int32), 0, self.receptive_field_size - 1)
        return box_coords_wrt_anchor

    def net_on_every_size(self, inputs_all_sizes, labels, filenames):

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
            grid.set_output_shape(net_shape[1], self.receptive_field_size)
            grid.set_flat_start_pos(self.n_boxes)
            self.max_pad_rel = max(self.max_pad_rel, grid.pad_rel)
            output_flat = tf.reshape(net, [-1, grid.n_boxes, CommonEncoding.get_last_layer_n_channels(self.opts, self.nclasses)], name='output_flat')
            cr_flat = tf.reshape(common_representation, [-1, grid.n_boxes, self.opts.lcr], name='cr_flat')
            self.n_boxes += grid.n_boxes
            all_outputs.append(output_flat)
            all_crs.append(cr_flat)
            if (grid.input_size_w_pad - self.receptive_field_size) / self.opts.step_in_pixels + 1 != grid.output_shape:
                raise Exception('Inconsistent step for input size ' + str(grid.input_size_w_pad) + '. Grid size is ' + str(grid.output_shape) + '.')
        assert self.n_boxes == self.get_expected_num_boxes(), 'Expected number of boxes differs from the real number.'
        all_outputs = tf.concat(all_outputs, axis=1, name='all_outputs')  # (batch_size, nboxes, ?)
        all_crs = tf.concat(all_crs, axis=1, name='all_crs')  # (batch_size, nboxes, lcr)
        self.input_image_size_w_pad = int(np.ceil(float(self.opts.input_image_size) / (1 - 2 * self.max_pad_rel)))
        self.max_pad_abs = int(np.round(self.max_pad_rel * self.input_image_size_w_pad))
        self.input_image_size_w_pad = self.opts.input_image_size + 2 * self.max_pad_abs
        print('')
        print('Maximum relative pad: ' + str(self.max_pad_rel))
        print('Maximum absolute pad: ' + str(self.max_pad_abs))
        print('Input image size with pad: ' + str(self.input_image_size_w_pad))
        self.print_grids_info()

        if self.opts.write_crops:
            inputs0 = inputs_all_sizes[0]
            inputs1 = inputs_all_sizes[1]
            inputs2 = inputs_all_sizes[2]
            labels_shape = labels.shape
            if len(inputs_all_sizes) == 1:
                labels = tf.py_func(self.write_input_crops_1grid, [inputs0, labels, filenames, all_outputs], (tf.float32))
            elif len(inputs_all_sizes) == 2:
                labels = tf.py_func(self.write_input_crops_2grids, [inputs0, inputs1, labels, filenames, all_outputs], (tf.float32))
            elif len(inputs_all_sizes) == 3:
                labels = tf.py_func(self.write_input_crops_3grids, [inputs0, inputs1, inputs2, labels, filenames, all_outputs], (tf.float32))
            else:
                raise Exception('Case with len(inputs_all_sizes) > 3 not implemented yet.')
            labels.set_shape(labels_shape)

        return all_outputs, all_crs, labels  # (batch_size, nboxes, ?)

    def compute_iou_of_two_anchors(self, index1, index2):
        coords1 = self.anchors_coordinates[index1, :]  # (4)
        coords2 = self.anchors_coordinates[index2, :]  # (4)
        iou = compute_iou_multi_dim(coords1, coords2)
        return iou

    def compare_all_crs(self, all_crs, labels, net_output_conf):
        # all_crs: (batch_size, nboxes, lcr)
        # labels: (batch_size, nboxes, 9)
        # net_output_conf: (batch_size, nboxes, nclasses)
        print('Making operations to compare CRs...')
        with tf.variable_scope('cr_comparison'):
            nearest_valid_gt_idx = labels[:, :, 7]  # (batch_size, nboxes)
            softmax = tf.nn.softmax(net_output_conf, axis=-1)  # (batch_size, nboxes, nclasses)
            max_softmax = tf.reduce_max(softmax, axis=-1)  # (batch_size, nboxes)
            self.comparisons_map = {}
            count = 0
            left_crs = []
            right_crs = []
            left_gt_idx = []
            right_gt_idx = []
            left_max_softmax = []
            right_max_softmax = []
            for i in range(self.n_boxes):
                for j in range(i + 1, self.n_boxes):
                    # We could set here some condition in order to compare only some boxes (IOU for instance).
                    if self.compute_iou_of_two_anchors(i, j) > self.opts.min_iou_to_compare:
                        left_crs.append(all_crs[:, i, :])  # (batch_size, lcr)
                        right_crs.append(all_crs[:, j, :])  # (batch_size, lcr)
                        left_gt_idx.append(nearest_valid_gt_idx[:, i])  # (batch_size)
                        right_gt_idx.append(nearest_valid_gt_idx[:, j])  # (batch_size)
                        left_max_softmax.append(max_softmax[:, i])  # (batch_size)
                        right_max_softmax.append(max_softmax[:, j])  # (batch_size)
                        self.comparisons_map[count] = [i, j]
                        count += 1
            self.n_comparisons = count
            print('Number of comparisons: ' + str(self.n_comparisons))
            print('Number of maximum possible comparisons: ' + str(int(scipy.misc.comb(self.n_boxes, 2))))
            assert self.n_comparisons > 0, 'Number of comparison must be greater than 0.'
            left_crs = tf.stack(left_crs, axis=1)  # (batch_size, n_comparisons, lcr)
            right_crs = tf.stack(right_crs, axis=1)  # (batch_size, n_comparisons, lcr)
            left_gt_idx = tf.stack(left_gt_idx, axis=-1)  # (batch_size, n_comparisons)
            right_gt_idx = tf.stack(right_gt_idx, axis=-1)  # (batch_size, n_comparisons)
            left_max_softmax = tf.stack(left_max_softmax, axis=-1)  # (batch_size, n_comparisons)
            right_max_softmax = tf.stack(right_max_softmax, axis=-1)  # (batch_size, n_comparisons)
            comparison_label = tf.less(tf.abs(left_gt_idx - right_gt_idx), 0.5)  # (batch_size, n_comparisons), bool
            subtraction = tf.abs(left_crs - right_crs)  # (batch_size, n_comparisons, lcr)
            subtraction = tf.expand_dims(subtraction, axis=2)  # (batch_size, n_comparisons, 1, lcr)
            fc1 = slim.conv2d(subtraction, self.opts.lcr, [1, 1], scope='fc1')  # (batch_size, n_comparisons, 1, lcr)
            comparison_pred = slim.conv2d(fc1, 2, [1, 1], scope='fc2', activation_fn=None)  # (batch_size, n_comparisons, 1, 2)
            comparison_pred = tf.squeeze(comparison_pred, axis=2)  # (batch_size, n_comparisons, 2)
            comp_conf1_wins = left_max_softmax > right_max_softmax  # (batch_size, n_comparisons)
        print('Operations to compare CRs ready.')
        return comparison_pred, comparison_label, comp_conf1_wins


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

    def make(self, inputs, labels, filenames):
        inputs_all_sizes = self.make_input_multiscale(inputs)
        net_output, all_crs, labels2 = self.net_on_every_size(inputs_all_sizes, labels, filenames)  # (batch_size, nboxes, ?)
        self.compute_anchors_coordinates()

        net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_tf(net_output, self.opts, self.nclasses)

        if self.opts.compare_boxes:
            comparison_pred, comparison_label, comp_conf1_wins = self.compare_all_crs(all_crs, labels, net_output_conf)
        else:
            comparison_pred = None
            comparison_label = None
            comp_conf1_wins = None

        if self.opts.write_crops:
            inputs0 = inputs_all_sizes[0]
            labels2_shape = labels2.shape
            labels2 = tf.py_func(self.write_predictions, [inputs0, labels, net_output_conf, filenames, net_output, labels2], (tf.float32))
            labels2.set_shape(labels2_shape)

        loss = self.make_loss(labels2, net_output, filenames, comparison_pred, comparison_label)

        if self.opts.compare_boxes:
            predictions = self.remove_repeated_predictions(net_output_conf, net_output_coords, net_output_pc, comparison_pred, comp_conf1_wins)
        else:
            predictions = net_output

        return predictions, loss

    def remove_repeated_predictions(self, net_output_conf, net_output_coords, net_output_pc, comparison_pred, comp_conf1_wins):
        # net_output: (batch_size, nboxes, ?)
        # net_output_conf: (batch_size, nboxes, nclasses)
        # comparison_pred: (batch_size, n_comparisons, 2)
        # comp_conf1_wins: (batch_size, n_comparisons)
        print('Setting operations to remove repeated predictions...')
        batch_size = tf.shape(net_output_conf)[0]
        background_conf = tf.concat([tf.zeros(shape=(batch_size, self.n_boxes, self.nclasses - 1), dtype=tf.float32),
                                     tf.ones(shape=(batch_size, self.n_boxes, 1), dtype=tf.float32)], axis=-1)
        predicted_same = comparison_pred[:, :, 1] > comparison_pred[:, :, 0]  # (batch_size, n_comparisons), bool
        all_masks = []
        for i in range(self.n_boxes):
            all_masks.append(tf.zeros((batch_size), tf.bool))
        for comp_idx in range(self.n_comparisons):
            index1 = self.comparisons_map[comp_idx][0]
            index2 = self.comparisons_map[comp_idx][1]
            supress_1 = tf.logical_and(predicted_same[:, comp_idx], tf.logical_not(comp_conf1_wins[:, comp_idx]))  # (batch_size)
            supress_2 = tf.logical_and(predicted_same[:, comp_idx], comp_conf1_wins[:, comp_idx])  # (batch_size)
            all_masks[index1] = tf.logical_or(all_masks[index1], supress_1)
            all_masks[index2] = tf.logical_or(all_masks[index2], supress_2)
        supression_mask = tf.stack(all_masks, axis=-1)  # (batch_size, nboxes)
        supression_mask = tf.tile(tf.expand_dims(supression_mask, axis=-1), [1, 1, self.nclasses])  # (batch_size, nboxes, nclasses)
        net_output_conf_mod = tf.where(supression_mask, background_conf, net_output_conf)
        predictions = CommonEncoding.put_together_net_output(net_output_conf_mod, net_output_coords, net_output_pc, self.opts, self.nclasses)
        print('Operations to remove repeated predictions ready.')
        return predictions

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

    def compute_indices_to_select_coords_and_pc(self, net_output_conf):
        # net_output_coords: (batch_size, nboxes, nclasses)
        # batch_size = net_output_conf.shape[0]
        batch_size = tf.shape(net_output_conf)[0]
        indices = tf.cast(tf.argmax(net_output_conf, axis=-1), tf.int32)  # (batch_size, nboxes)
        indices_mesh = tf.stack(tf.meshgrid(tf.range(0, self.n_boxes), tf.range(0, batch_size)) + [indices], axis=2)  # (batch_size, nboxes, 3)
        print('indices_mesh: ' + str(indices_mesh))
        return indices_mesh

    def compute_localization_and_pc_losses(self, mask_match, labels_enc, net_output_conf, net_output_coords, net_output_pc, n_positives, zeros):
        # mask_match: (batch_size, nboxes)
        gt_pc_ass = labels_enc[:, :, 8]  # (batch_size, nboxes)
        gt_coords = labels_enc[:, :, :4]  # (batch_size, nboxes, 4)
        if self.opts.box_per_class:
            # net_output_coords: (batch_size, nboxes, 4 * nclasses)
            # [dcx_enc_cl1, dcx_enc_cl2, ..., dcy_enc_cl1, dcy_enc_cl2, ..., w_enc_cl1, w_enc_cl2, ..., h_enc_cl1, h_enc_cl2, ...]
            indices_mesh = self.compute_indices_to_select_coords_and_pc(net_output_conf)
            selected_coords = CommonEncoding.get_selected_coords_tf(indices_mesh, net_output_coords, self.nclasses)  # (batch_size, nboxes, 4)

            localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, selected_coords)  # (batch_size, nboxes)
            # print('localization_loss = ' + str(localization_loss))

            if self.opts.predict_pc:
                # net_output_pc: (batch_size, nboxes, nclasses)
                selected_pc = tf.gather_nd(net_output_pc, indices_mesh)  # (batch_size, nboxes)
                pc_loss = CommonEncoding.smooth_L1_loss(gt_pc_ass, selected_pc, False)  # (batch_size, nboxes)
        else:
            # pred_cords: (batch_size, nboxes, 4)
            localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, net_output_coords)  # (batch_size, nboxes)
            # print('localization_loss = ' + str(localization_loss))

            if self.opts.predict_pc:
                # net_output_pc: (batch_size, nboxes)
                pc_loss = CommonEncoding.smooth_L1_loss(gt_pc_ass, net_output_pc, False)  # (batch_size, nboxes)

        # localization_loss_matches = tf.reduce_sum(localization_loss * mask_match, axis=-1)  # (batch_size)
        localization_loss_matches = tf.where(mask_match, localization_loss, zeros, name='loc_loss_match')  # (batch_size)
        loss_loc_scaled = tf.divide(localization_loss_matches, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_loc_scaled')
        loss_loc = tf.reduce_sum(loss_loc_scaled, name='loss_loc')

        if self.opts.predict_pc:
            # pc_loss_matches = tf.reduce_sum(pc_loss * mask_match, axis=-1)  # (batch_size)
            pc_loss_matches = tf.where(mask_match, pc_loss, zeros, name='pc_loss_match')  # (batch_size)
            loss_pc_scaled = tf.divide(pc_loss_matches, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_pc_scaled')
            loss_pc = tf.reduce_sum(loss_pc_scaled, name='loss_pc')
            return loss_loc, loss_pc
        else:
            return loss_loc

    def compute_conf_loss(self, mask_match, mask_neutral, net_output_conf, labels, n_positives, zeros):

        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=net_output_conf, name='loss_orig')
        # loss_orig = tf.Print(loss_orig, [tf.reduce_sum(loss_orig)], 'loss_orig')
        # loss_pos = mask_match * loss_orig
        loss_pos = tf.where(mask_match, loss_orig, zeros, name='loss_pos')
        # loss_pos = tf.Print(loss_pos, [tf.reduce_sum(loss_pos)], 'loss_pos')

        # mask_match = tf.Print(mask_match, [n_positives], 'n_positives')
        # mask_neutral = tf.Print(mask_neutral, [n_neutrals], 'n_neutrals')
        mask_negatives = tf.logical_and(tf.logical_not(mask_match, name='not_mask_match'), tf.logical_not(mask_neutral, name='not_mask_neutral'), name='mask_negatives')
        n_negatives = tf.reduce_sum(tf.cast(mask_negatives, tf.int32, name='mask_negatives_int32'), name='n_negatives')
        # mask_negatives = tf.Print(mask_negatives, [n_negatives], 'n_negatives')
        # loss_neg = mask_negatives * loss_orig
        loss_neg = tf.where(mask_negatives, loss_orig, zeros, name='loss_neg')
        # loss_neg = tf.Print(loss_neg, [tf.reduce_sum(loss_neg)], 'loss_neg')

        loss_pos_scaled = tf.divide(loss_pos, tf.maximum(tf.cast(n_positives, tf.float32, name='n_positives_float32'), 1, name='max_n_positives_1'), name='loss_pos_scaled')
        loss_neg_scaled = tf.divide(loss_neg, tf.maximum(tf.cast(n_negatives, tf.float32, name='n_negatives_float32'), 1, name='max_n_negatives_1'), name='loss_neg_scaled')
        # loss_pos_scaled = tf.Print(loss_pos_scaled, [tf.reduce_sum(loss_pos_scaled)], 'loss_pos_scaled')
        # loss_neg_scaled = tf.Print(loss_neg_scaled, [tf.reduce_sum(loss_neg_scaled)], 'loss_neg_scaled')
        # tf.summary.scalar("loss_pos_scaled", loss_pos_scaled)
        # tf.summary.scalar("loss_neg_scaled", loss_neg_scaled)

        loss_conf = tf.reduce_sum(loss_pos_scaled + loss_neg_scaled, name='loss_conf')

        return loss_conf

    def comparison_loss(self, comparison_pred, comparison_label):
        # comparison_pred: (batch_size, n_comarisons, 2)
        # comparison_pred: (batch_size, n_comarisons)
        comparison_label_int = tf.cast(comparison_label, tf.int32)
        comp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=comparison_label_int, logits=comparison_pred, name='comp_loss_orig')
        comp_loss = tf.reduce_mean(comp_loss) * self.opts.comp_loss_factor
        return comp_loss

    def make_loss(self, labels_enc, net_output, filenames, comparison_pred, comparison_label):
        # Inputs:
        #     encoded_labels: numpy array of dimension (batch_size, nboxes x 9), being nboxes the total number of anchor boxes.
        #                     The second dimension is like follows: First the 4 encoded coordinates of the bounding box,
        #                     then a flag indicating if it matches a ground truth box or not, another flag inidicating
        #                     if it is a neutral box, the class id, the ground truth box id associated with, and then
        #                     the maximum percent contained (or the percent contained of the associated gt box)
        print('Defining loss...')
        with tf.name_scope('myloss'):
            mask_match = labels_enc[:, :, 4]  # (batch_size, nboxes)
            mask_neutral = labels_enc[:, :, 5]  # (batch_size, nboxes)
            gt_class_ids = labels_enc[:, :, 6]  # (batch_size, nboxes)
            nearest_valid_gt_idx = labels_enc[:, :, 7]  # (batch_size, nboxes)

            net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_tf(net_output, self.opts, self.nclasses)

            labels = tf.cast(tf.round(gt_class_ids, name='round_gt_class_ids'), tf.int32, name='labels_int32')

            mask_match = tf.greater(mask_match, 0.5, name='mask_match')
            mask_neutral = tf.greater(mask_neutral, 0.5, name='mask_neutral')

            n_positives = tf.reduce_sum(tf.cast(mask_match, tf.int32, name='mask_match_int32'), name='n_positives')
            n_neutrals = tf.reduce_sum(tf.cast(mask_neutral, tf.int32, name='mask_neutral_int32'), name='n_neutrals')

            zeros = tf.zeros(shape=tf.shape(mask_match, name='mask_match_shape'), dtype=tf.float32, name='zeros_mask')

            # Confidence loss:
            loss_conf = self.compute_conf_loss(mask_match, mask_neutral, net_output_conf, labels, n_positives, zeros)
            tf.summary.scalar("loss_conf", loss_conf)
            # print('loss_conf')
            # print(loss_conf)

            # loss_conf = tf.Print(loss_conf, [loss_conf], 'loss_conf')

            total_loss = loss_conf

            if self.opts.predict_coordinates:
                if self.opts.predict_pc:
                    loc_loss, pc_loss = self.compute_localization_and_pc_losses(mask_match, labels_enc,
                                                                net_output_conf, net_output_coords,
                                                                net_output_pc, n_positives, zeros)
                    tf.summary.scalar("loc_loss", loc_loss)
                    tf.summary.scalar("pc_loss", pc_loss)
                    # loc_loss = tf.Print(loc_loss, [loc_loss], 'loc_loss')
                    # pc_loss = tf.Print(pc_loss, [pc_loss], 'pc_loss')
                    total_loss += loc_loss * self.opts.loc_loss_factor
                    total_loss += pc_loss * self.opts.pc_loss_factor
                else:
                    loc_loss = self.compute_localization_and_pc_losses(mask_match, labels_enc,
                                                    net_output_conf, net_output_coords,
                                                    net_output_pc, n_positives, zeros)
                    tf.summary.scalar("loc_loss", loc_loss)
                    # loc_loss = tf.Print(loc_loss, [loc_loss], 'loc_loss')
                    total_loss += loc_loss * self.opts.loc_loss_factor

            if self.opts.compare_boxes:
                loss_comp = self.comparison_loss(comparison_pred, comparison_label)
                tf.summary.scalar("loss_comp", loss_comp)
                total_loss += loss_comp

            tf.summary.scalar("total_loss", total_loss)

        print('Loss defined.')
        return total_loss

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

    def encode_gt(self, gt_boxes, filename):
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

        filename_dec = filename.decode(sys.getdefaultencoding())
        # print(filename_dec)

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

            # gt_vec = tf.Print(gt_vec, [tf.shape(gt_vec)], 'tf.shape(gt_vec)')

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

            # mask_thresholds = tf.Print(mask_thresholds, [tf.shape(mask_thresholds)], 'tf.shape(mask_thresholds)')

            dc_masked = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt, self.n_boxes), dtype=np.float32) ) # (n_gt, n_boxes)

            # dc_masked = tf.Print(dc_masked, [tf.shape(dc_masked)], 'tf.shape(dc_masked)')
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

            if self.opts.write_pc_ar_dc:
                self.write_pc_ar_dc_pcass(pc, ar, dc, pc_associated, filename_dec)

            # if (np.abs(pc_associated - 1) < 1e-4):
            #     print('pc_associated = ' + str(pc_associated))
            # pc_associated_enc = logit(pc_associated)
            pc_associated_enc = pc_associated

            # Put all together in one array:
            labels_enc = coordinates_enc  # (nboxes, 4)
            labels_enc = np.concatenate((labels_enc, np.expand_dims(mask_match.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 5)  pos 4
            labels_enc = np.concatenate((labels_enc, np.expand_dims(mask_neutral.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 6)  pos 5
            labels_enc = np.concatenate((labels_enc, np.expand_dims(class_ids.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 7)  pos 6
            labels_enc = np.concatenate((labels_enc, np.expand_dims(nearest_valid_gt_idx.astype(np.float32), axis=1)), axis=1)  # (n_boxes, 8)  pos 7
            labels_enc = np.concatenate((labels_enc, np.expand_dims(pc_associated_enc, axis=1)), axis=1)  # (n_boxes, 9)  pos 8

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

        # self.anchors_coordinates (nboxes, 4) [xmin, ymin, xmax, ymax]
        anchors_xc = (self.anchors_coordinates[:, 0] + self.anchors_coordinates[:, 2]) / 2.0
        anchors_yc = (self.anchors_coordinates[:, 1] + self.anchors_coordinates[:, 3]) / 2.0
        anchors_w = self.anchors_coordinates[:, 2] - self.anchors_coordinates[:, 0]
        anchors_h = self.anchors_coordinates[:, 3] - self.anchors_coordinates[:, 1]

        dcx = (anchors_xc - xc) / (anchors_w * 0.5)  # (nboxes)
        dcy = (anchors_yc - yc) / (anchors_h * 0.5)  # (nboxes)
        # Between -1 and 1 for the box to lie inside the anchor.

        w_rel = width / anchors_w  # (nboxes)
        h_rel = height / anchors_h  # (nboxes)
        # Between 0 and 1 for the box to lie inside the anchor.

        # Encoding step:
        if self.opts.encoding_method == 'basic_1':
            dcx_enc = np.tan(dcx * (np.pi / 2.0 - self.opts.enc_epsilon))
            dcy_enc = np.tan(dcy * (np.pi / 2.0 - self.opts.enc_epsilon))
            w_enc = CommonEncoding.((w_rel - self.opts.enc_wh_b) / self.opts.enc_wh_a)
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
            w_rel = np.clip(CommonEncoding.sigmoid(w_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / self.receptive_field_size, 1.0)
            h_rel = np.clip(CommonEncoding.sigmoid(h_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / self.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'ssd':
            dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)
            w_rel = np.clip(np.exp(w_enc * 0.2), 1.0 / self.receptive_field_size, 1.0)
            h_rel = np.clip(np.exp(h_enc * 0.2), 1.0 / self.receptive_field_size, 1.0)
        elif self.opts.encoding_method == 'no_encode':
            dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
            w_rel = np.clip(w_enc, 1.0 / self.receptive_field_size, 1.0)
            h_rel = np.clip(h_enc, 1.0 / self.receptive_field_size, 1.0)
        else:
            raise Exception('Encoding method not recognized.')

        # self.anchors_coordinates (nboxes, 4) [xmin, ymin, xmax, ymax]
        anchors_xc = (self.anchors_coordinates[..., 0] + self.anchors_coordinates[..., 2]) / 2.0
        anchors_yc = (self.anchors_coordinates[..., 1] + self.anchors_coordinates[..., 3]) / 2.0
        anchors_w = self.anchors_coordinates[..., 2] - self.anchors_coordinates[..., 0]
        anchors_h = self.anchors_coordinates[..., 3] - self.anchors_coordinates[..., 1]

        xc = anchors_xc - dcx_rel * (anchors_w * 0.5)  # (nboxes)
        yc = anchors_yc - dcy_rel * (anchors_h * 0.5)  # (nboxes)

        width = anchors_w * w_rel  # (nboxes)
        height = anchors_h * h_rel  # (nboxes)

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
            w_rel = np.clip(CommonEncoding.sigmoid(w_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / self.receptive_field_size, 1.0)  # (batch_size, nboxes)
            h_rel = np.clip(CommonEncoding.sigmoid(h_enc) * self.opts.enc_wh_a + self.opts.enc_wh_b, 1.0 / self.receptive_field_size, 1.0)  # (batch_size, nboxes)
        elif self.opts.encoding_method == 'ssd':
            dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)  # (batch_size, nboxes)
            dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)  # (batch_size, nboxes)
            w_rel = np.clip(np.exp(w_enc * 0.2), 1.0 / self.receptive_field_size, 1.0)  # (batch_size, nboxes)
            h_rel = np.clip(np.exp(h_enc * 0.2), 1.0 / self.receptive_field_size, 1.0)  # (batch_size, nboxes)
        elif self.opts.encoding_method == 'no_encode':
            dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
            dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
            w_rel = np.clip(w_enc, 1.0 / self.receptive_field_size, 1.0)
            h_rel = np.clip(h_enc, 1.0 / self.receptive_field_size, 1.0)
        else:
            raise Exception('Encoding method not recognized.')

        # self.anchors_coordinates (nboxes, 4) [xmin, ymin, xmax, ymax]
        anchors_coords_exp = np.expand_dims(self.anchors_coordinates, axis=0)  # (1, nboxes, 4)
        batch_size = coords_enc.shape[0]
        anchors_coords_exp = np.tile(anchors_coords_exp, [batch_size, 1, 1])  # (batch_size, nboxes, 4)
        anchors_xc = (anchors_coords_exp[:, :, 0] + anchors_coords_exp[:, :, 2]) / 2.0  # (batch_size, nboxes)
        anchors_yc = (anchors_coords_exp[:, :, 1] + anchors_coords_exp[:, :, 3]) / 2.0  # (batch_size, nboxes)
        anchors_w = anchors_coords_exp[:, :, 2] - anchors_coords_exp[:, :, 0]  # (batch_size, nboxes)
        anchors_h = anchors_coords_exp[:, :, 3] - anchors_coords_exp[:, :, 1]  # (batch_size, nboxes)

        xc = anchors_xc - dcx_rel * (anchors_w * 0.5)  # (batch_size, nboxes)
        yc = anchors_yc - dcy_rel * (anchors_h * 0.5)  # (batch_size, nboxes)

        width = anchors_w * w_rel  # (batch_size, nboxes)
        height = anchors_h * h_rel  # (batch_size, nboxes)

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
        ar = area_gt_sq / area_anchors  #  n_gt, nboxes)

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

    def compute_repeated_preds_metric(self, predictions, labels_enc):
        # predictions: (batch_size, nboxes, ?)
        # labels_enc: (batch_size, nboxes, 9)
        batch_size = labels_enc.shape[0]
        n_repetitions = 0
        n_gt = 0
        for b in range(batch_size):
            net_output_nobatch = predictions[b, ...]  # (nboxes, ?)
            net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(net_output_nobatch, self.opts, self.nclasses)
            # net_output_conf: (nboxes, nclasses)
            nearest_valid_gt_idx = labels_enc[b, :, 7]  # (nboxes)
            winning_class = np.argmax(net_output_conf, axis=-1)  # (nboxes)
            is_background = np.less(np.abs(winning_class - self.background_id), 0.5)  # (nboxes)
            gt_count = {}
            for i in range(self.n_boxes):
                if nearest_valid_gt_idx[i] in gt_count:
                    if not is_background[i]:
                        gt_count[nearest_valid_gt_idx[i]] += 1
                        n_repetitions += 1
                else:
                    n_gt += 1
                    if not is_background[i]:
                        gt_count[nearest_valid_gt_idx[i]] = 1
        # proportion_repetitions = float(n_repetitions) / n_gt
        proportion_repetitions = float(n_repetitions) / (self.n_boxes * batch_size)
        metric_repetitions = 1 - proportion_repetitions
        return metric_repetitions

    def compute_class_accuracy(self, net_output, labels_enc):
        # net_output: (batch_size, nboxes, ?)
        # labels_enc: (batch_size, nboxes, 9)
        n_hits = 0
        batch_size = labels_enc.shape[0]
        for b in range(batch_size):
            net_output_nobatch = net_output[b, ...]  # (nboxes, ?)
            net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(net_output_nobatch, self.opts, self.nclasses)
            # net_output_conf: (nboxes, nclasses)
            hits = np.abs(np.argmax(net_output_conf, axis=-1) - labels_enc[b, :, 6]) < 0.5
            n_hits += np.sum(hits.astype(np.int32))
        n_attemps = batch_size * self.n_boxes
        accuracy = float(n_hits) / n_attemps
        return accuracy

    def compute_iou_metric(self, net_output, labels_enc):
        # net_output: (batch_size, nboxes, ?)
        # labels_enc: (batch_size, nboxes, 9)
        mask_match = labels_enc[:, :, 4]  # (batch_size, nboxes)
        gt_class_ids = np.round(labels_enc[:, :, 6]).astype(np.int32)  # (batch_size, nboxes)
        net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(net_output, self.opts, self.nclasses)
        selected_coords = CommonEncoding.get_selected_coords_np(net_output_conf, net_output_coords, self.nclasses, self.opts.box_per_class)  # (batch_size, nboxes, 4)
        pred_class_ids = np.argmax(net_output_conf, axis=-1)  # (batch_size, nboxes)
        mask_TP = np.equal(gt_class_ids, pred_class_ids) * mask_match  # (batch_size, nboxes)
        n_matches = np.sum(mask_TP)  # ()
        if n_matches > 0.5:
            pred_coords = self.decode_boxes_w_batch_size(selected_coords)  # (batch_size, nboxes, 4) [xmin, ymin, width, height]
            gt_coords = self.decode_boxes_w_batch_size(labels_enc[:, :, :4])  # (batch_size, nboxes, 4) [xmin, ymin, width, height]
            iou = compute_iou_multi_dim(gt_coords, pred_coords)  # (batch_size, nboxes)
            iou_on_match = mask_TP * iou  # (batch_size, nboxes)
            mean_iou = np.sum(iou_on_match) / n_matches  # ()
            # for i in range(iou.shape[0]):
            #     if mask_match[i, 0] > 0.5:
            #         print(iou[i, 0])
            # print('compute_iou_metric: ' + str(mean_iou))
        else:
            mean_iou = None
            # print('compute_iou_metric: None')
        return mean_iou

    def write_pc_ar_dc_pcass(self, pc, ar, dc, pc_associated, filename):
        n_gt = pc.shape[0]
        if not os.path.exists(self.input_crops_dir):
            os.makedirs(self.input_crops_dir)
        file_path = os.path.join(self.input_crops_dir, filename + '_pc_ar_dc_pcass.csv')
        print('pc_ar_dc_pcass file path: ' + file_path)
        with open(file_path, 'w') as fid:
            for i in range(self.n_boxes):
                fid.write('\n')
                fid.write('Box ' + str(i) + '\n')
                fid.write('pc: ')
                for j in range(n_gt):
                    fid.write(';' + str(pc[j, i]))
                fid.write('\n')
                fid.write('ar: ')
                for j in range(n_gt):
                    fid.write(';' + str(ar[j, i]))
                fid.write('\n')
                fid.write('dc: ')
                for j in range(n_gt):
                    fid.write(';' + str(dc[j, i]))
                fid.write('\n')
                fid.write('pc_associated: ')
                fid.write(';' + str(pc_associated[i]) + '\n')




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




