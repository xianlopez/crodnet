import os
import tensorflow as tf
import network
import numpy as np
import CommonEncoding
import tools
from BoundingBoxes import BoundingBox, PredictedBox
import cv2
import sys


class SingleCellArch:
    def __init__(self, options, nclasses, outdir=None):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.n_labels = 9
        self.batch_size = self.opts.n_images_per_batch * self.opts.n_crops_per_image
        self.metric_names = ['accuracy_conf', 'iou_mean', 'accuracy_comp']
        self.n_metrics = len(self.metric_names)
        self.outdir = outdir
        self.classnames = None
        assert self.opts.n_images_per_batch >= 2, 'n_images_per_batch must be at least 2.'
        if (self.opts.debug_train or self.opts.debug_eval) and self.outdir is None:
            raise Exception('outdir must be specified if debugging is True.')
        if self.opts.debug_eval:
            self.create_folders_for_eval_debug()
        self.proportion_valid_comparisons = 0
        self.proportion_same_comparisons = 0
        self.batch_count_train_debug = 0
        self.batch_count_eval_debug = 0

    def set_classnames(self, classnames):
        print('Setting classnames')
        self.classnames = []
        for name in classnames:
            self.classnames.append(name)
        self.classnames.append('background')
        print('classnames')
        print(self.classnames)

    def make(self, inputs, labels_enc, filenames):
        # inputs: (n_images_per_batch, n_crops_per_image, input_image_size, input_image_size, 3)
        # labels_enc: (n_images_per_batch, n_crops_per_image, n_labels)
        # filenames: (n_images_per_batch)
        inputs_reord = simplify_batch_dimensions(inputs)  # (batch_size, input_image_size, input_image_size, 3)
        labels_enc_reord = simplify_batch_dimensions(labels_enc)  # (batch_size, n_labels)
        common_representation = network.common_representation(inputs_reord, self.opts.lcr)  # (batch_size, 1, 1, lcr)
        loc_and_classif = network.localization_and_classification_path(common_representation, self.opts, self.nclasses)  # (batch_size, 1, 1, 4+nclasses)
        common_representation = tf.squeeze(common_representation, axis=[1, 2])  # (batch_size, lcr)
        loc_and_classif = tf.squeeze(loc_and_classif, axis=[1, 2])  # (batch_size, 4+nclasses)
        comparisons_pred, comps_validity_labels, comparisons_indices = self.make_comparisons(common_representation, labels_enc_reord)
        # comparisons_pred, indices_all_comparisons: (total_comparisons, 2)
        # comparisons_labels: (total_comparisons)
        loss, metrics = self.make_loss_and_metrics(loc_and_classif, labels_enc_reord, comparisons_pred, comps_validity_labels)  # ()
        if self.opts.debug_train:
            metrics = tf.py_func(self.write_train_debug_info, [metrics, inputs_reord, labels_enc_reord, loc_and_classif,
                                                         comparisons_pred, comps_validity_labels, comparisons_indices, filenames], (tf.float32))
        if self.opts.debug_eval:
            metrics = tf.py_func(self.write_eval_debug_info, [metrics, inputs_reord, labels_enc_reord, loc_and_classif,
                                                         comparisons_pred, comps_validity_labels, comparisons_indices, filenames], (tf.float32))

        return loc_and_classif, loss, metrics

    def make_comparisons(self, common_representation, labels_enc_reord):
        # common_representation: (batch_size, lcr)
        # labels_enc_reord: (batch_size, n_labels)
        gt_class_id = CommonEncoding.get_gt_class(labels_enc_reord)  # (batch_size)
        nearest_valid_gt_idx = CommonEncoding.get_nearest_valid_gt_idx(labels_enc_reord)  # (batch_size)
        mask_neutral = CommonEncoding.get_mask_neutral(labels_enc_reord)  # (batch_size)
        images_range = tf.range(self.opts.n_images_per_batch)  # (n_images_per_batch)

        # Indices of intra comprarisons:
        total_comparisons_intra = self.opts.n_images_per_batch * self.opts.n_comparisons_intra
        image_indices_comp_intra = tf.tile(tf.expand_dims(images_range, axis=-1), [1, self.opts.n_comparisons_intra])  # (n_images_per_batch, n_comparisons_intra)
        image_indices_comp_intra = tf.reshape(image_indices_comp_intra, [total_comparisons_intra])  # (total_comparisons_intra)
        # image_indices_comp_intra: [0, 0, ..., 0, 1, 1, ..., 1, ..., n_images_per_batch, n_images_per_batch, ..., n_images_per_batch]
        image_indices_comp_intra_exp = tf.tile(tf.expand_dims(image_indices_comp_intra, axis=-1), [1, 2])  # (total_comparisons_intra, 2)
        # Generate random indices relative to their image:
        crop_indices_intra_left = tf.random_uniform(shape=[total_comparisons_intra], maxval=self.opts.n_crops_per_image, dtype=tf.int32)
        crop_indices_distance = tf.random_uniform(shape=[total_comparisons_intra], minval=1, maxval=self.opts.n_crops_per_image, dtype=tf.int32)
        crop_indices_intra_right = tf.floormod(crop_indices_intra_left + crop_indices_distance, self.opts.n_crops_per_image)
        crop_indices_intra = tf.stack([crop_indices_intra_left, crop_indices_intra_right], axis=-1)  # (total_comparisons_intra, 2)
        # Convert this relative indices to absolute:
        random_indices_intra = crop_indices_intra + self.opts.n_crops_per_image * image_indices_comp_intra_exp  # (total_comparisons_intra, 2)

        # Indices of inter comparisons:
        total_comparisons_inter = self.opts.n_images_per_batch * self.opts.n_comparisons_inter
        crop_indices_inter = tf.random_uniform(shape=(total_comparisons_inter, 2), maxval=self.opts.n_crops_per_image, dtype=tf.int32)
        image_indices_comp_inter_left = tf.random_uniform(shape=[total_comparisons_inter], maxval=self.opts.n_images_per_batch, dtype=tf.int32)
        image_indices_distance = tf.random_uniform(shape=[total_comparisons_inter], minval=1, maxval=self.opts.n_images_per_batch, dtype=tf.int32)
        image_indices_comp_inter_right = tf.floormod(image_indices_comp_inter_left + image_indices_distance, self.opts.n_images_per_batch)
        image_indices_comp_inter = tf.stack([image_indices_comp_inter_left, image_indices_comp_inter_right], axis=-1)  # (total_comparisons_inter, 2)
        random_indices_inter = crop_indices_inter + self.opts.n_crops_per_image * image_indices_comp_inter  # (total_comparisons_inter, 2)

        # CRs of comparisons:
        indices_all_comparisons = tf.concat([random_indices_intra, random_indices_inter], axis=0)  # (total_comparisons, 2)
        crs_all_comparisons = tf.gather(common_representation, indices_all_comparisons, axis=0)  # (total_comparisons, 2, lcr)

        # Comparisons:
        comparisons_pred = network.comparison(crs_all_comparisons, self.opts.lcr)  # (total_comparisons, 2)

        # Labels of comparisons:
        gt_idx_intra_comp = tf.gather(nearest_valid_gt_idx, random_indices_intra, axis=0)  # (total_comparisons_intra, 2)
        same_gt_idx = tf.less(tf.abs(gt_idx_intra_comp[:, 0] - gt_idx_intra_comp[:, 1]), 0.5)  # (total_comparisons_intra)
        gt_class_intra_comp = tf.gather(gt_class_id, random_indices_intra, axis=0)  # (total_comparisons_intra, 2)
        any_background = tf.less(tf.abs(gt_class_intra_comp - self.background_id), 0.5)  # (total_comparisons_intra, 2)
        any_background = tf.reduce_any(any_background, axis=1)  # (total_comparisons_intra)
        any_neutral = tf.gather(mask_neutral, random_indices_intra, axis=0)  # (total_comparisons_intra, 2)
        any_neutral = tf.greater(any_neutral, 0.5)
        any_neutral = tf.reduce_any(any_neutral, axis=1)  # (total_comparisons_intra)
        valid_comps_intra = tf.logical_not(tf.logical_or(any_background, any_neutral))
        labels_comp_intra = same_gt_idx  # (total_comparisons_intra)
        labels_comp_inter = tf.zeros(shape=(total_comparisons_inter), dtype=tf.bool)
        valid_comps_inter = tf.ones(shape=(total_comparisons_inter), dtype=tf.bool)
        labels_all_comparisons = tf.concat([labels_comp_intra, labels_comp_inter], axis=0)  # (total_comparisons)
        valididty_all_comparisons = tf.concat([valid_comps_intra, valid_comps_inter], axis=0)  # (total_comparisons)
        validity_labels_comps = tf.stack([valididty_all_comparisons, labels_all_comparisons], axis=1)  # (total_comparisons, 2)

        return comparisons_pred, validity_labels_comps, indices_all_comparisons

    def make_loss_and_metrics(self, loc_and_classif, labels_enc_reord, comparisons_pred, comps_validity_labels):
        # common_representation: (batch_size, lcr)
        # loc_and_classif: (batch_size, 4+nclasses)
        # labels_enc_reord: (batch_size, n_labels)
        # comparisons_pred: (total_comparisons, 2)
        # comps_validity_labels: (total_comparisons, 2)  [validity, label]

        pred_conf = loc_and_classif[:, 4:]  # (batch_size, nclasses)
        pred_coords = loc_and_classif[:, :4]  # (batch_size, 4)

        mask_match = CommonEncoding.get_mask_match(labels_enc_reord)  # (batch_size)
        mask_neutral = CommonEncoding.get_mask_neutral(labels_enc_reord)  # (batch_size)
        gt_class_ids = CommonEncoding.get_gt_class(labels_enc_reord)  # (batch_size)
        gt_coords = CommonEncoding.get_gt_coords(labels_enc_reord)  # (batch_size)

        mask_match = tf.greater(mask_match, 0.5)
        mask_neutral = tf.greater(mask_neutral, 0.5)

        zeros = tf.zeros_like(mask_match, dtype=tf.float32)  # (batch_size)
        n_positives = tf.reduce_sum(tf.cast(mask_match, tf.int32), name='n_positives')  # ()

        conf_loss, accuracy_conf = classification_loss_and_metric(pred_conf, mask_match, mask_neutral,
                                                                  gt_class_ids, zeros, n_positives)
        tf.summary.scalar('losses/conf_loss', conf_loss)
        tf.summary.scalar('metrics/accuracy_conf', accuracy_conf)
        total_loss = conf_loss

        loc_loss, iou_mean = localization_loss_and_metric(pred_coords, mask_match, mask_neutral, gt_coords, zeros,
                                                          self.opts.loc_loss_factor, self.opts)
        tf.summary.scalar('losses/loc_loss', loc_loss)
        tf.summary.scalar('metrics/iou_mean', iou_mean)
        total_loss += loc_loss

        comp_loss, accuracy_comp = comparison_loss_and_metric(comparisons_pred, comps_validity_labels, self.opts.comp_loss_factor)
        tf.summary.scalar('losses/comp_loss', comp_loss)
        tf.summary.scalar('metrics/accuracy_comp', accuracy_comp)
        total_loss += comp_loss

        metrics = tf.stack([accuracy_conf, iou_mean, accuracy_comp])  # (n_metrics)

        return total_loss, metrics

    def get_input_shape(self):
        input_shape = [network.receptive_field_size, network.receptive_field_size]
        return input_shape

    def encode_gt_from_array(self, gt_boxes):
        # gt_boxes: (n_gt, 7) [class_id, xmin, ymin, width, height, pc, gt_idx]
        n_gt = gt_boxes.shape[0]
        if n_gt > 0:
            gt_coords = gt_boxes[:, 1:5]  # (n_gt, 4)
            gt_class_ids = gt_boxes[:, 0]  # (n_gt)
            gt_pc_incoming = gt_boxes[:, 5]  # (n_gt)

            ar, dc = compute_ar_dc(gt_coords)  # All variables have shape (n_gt)
            pc = gt_pc_incoming  # (n_gt)

            # Positive boxes:
            mask_ar = ar > self.opts.threshold_ar  # (n_gt)
            mask_pc = pc > self.opts.threshold_pc  # (n_gt)
            mask_dc = dc < self.opts.threshold_dc  # (n_gt)
            mask_thresholds = mask_ar & mask_pc & mask_dc  # (n_gt)
            any_match = np.any(mask_thresholds)  # ()

            # Neutral boxes:
            mask_ar_neutral = ar > self.opts.threshold_ar_neutral  # (n_gt)
            mask_pc_neutral = pc > self.opts.threshold_pc_neutral  # (n_gt)
            mask_dc_neutral = dc < self.opts.threshold_dc_neutral  # (n_gt)
            mask_thresholds_neutral = mask_ar_neutral & mask_pc_neutral & mask_dc_neutral  # (n_gt)
            any_neutral = np.any(mask_thresholds_neutral, axis=0)  # ()
            is_neutral = np.logical_and(any_neutral, np.logical_not(any_match))  # ()

            if any_match:
                dc_masked = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt), dtype=np.float32) ) # (n_gt)
            else:
                dc_masked = np.where(mask_thresholds_neutral, dc, np.infty * np.ones(shape=(n_gt), dtype=np.float32) ) # (n_gt)

            nearest_valid_box_idx = np.argmin(dc_masked)  # ()

            # Get the coordinates and the class id of the gt box matched:
            coordinates = gt_coords[nearest_valid_box_idx, :]  # (4)
            coordinates_enc = encode_boxes_np(coordinates, self.opts)  # (4)
            class_id = gt_class_ids[nearest_valid_box_idx]

            # Negative boxes:
            is_negative = np.logical_and(np.logical_not(any_match), np.logical_not(is_neutral))  # ()
            if is_negative:
                class_id = self.background_id

            # Percent contained associated with each anchor box.
            # This is the PC of the assigned gt box, if there is any, or otherwise the maximum PC it has.
            if any_match:
                pc_associated = pc[nearest_valid_box_idx]
            else:
                pc_associated = np.max(pc)

            # Take the original GT index:
            associated_gt_idx = gt_boxes[nearest_valid_box_idx, 6]  # ()

            # Put all together in one array:
            labels_enc = np.stack([any_match.astype(np.float32),
                                   is_neutral.astype(np.float32),
                                   float(class_id),
                                   associated_gt_idx,
                                   pc_associated])
            labels_enc = np.concatenate([coordinates_enc, labels_enc])  # (size_labels_enc)

        else:
            labels_enc = np.zeros(shape=(9), dtype=np.float32)
            labels_enc[6] = self.background_id

        return labels_enc  # (9)

    def write_train_debug_info(self, metrics, inputs_reord, labels_enc_reord, loc_and_classif, comparisons_pred, comps_validity_labels, comparisons_indices, filenames):
        # inputs_reord: (batch_size, input_image_size, input_image_size, 3)
        # labels_enc_reord: (batch_size, n_labels)
        # loc_and_classif: (batch_size, 4+nclasses)
        # comparisons_pred: (total_comparisons, 2)
        # comps_validity_labels: (total_comparisons, 2)  [validity, label]
        # comparisons_indices: (total_comparisons, 2)
        # filenames: (n_images_per_batch)
        if self.classnames is None:
            raise Exception('classnames must be specified if debugging.')
        self.batch_count_train_debug += 1
        batch_dir = os.path.join(self.outdir, 'batch' + str(self.batch_count_train_debug))
        os.makedirs(batch_dir)
        batch_size = self.opts.n_crops_per_image * self.opts.n_images_per_batch
        filenames_ext = np.tile(np.expand_dims(filenames, axis=-1), [1, self.opts.n_crops_per_image])  # (n_images_per_batch, n_crops_per_image)
        filenames_reord = np.reshape(filenames_ext, newshape=(batch_size))  # (batch_size)
        mask_match = CommonEncoding.get_mask_match(labels_enc_reord) > 0.5  # (batch_size)
        mask_neutral = CommonEncoding.get_mask_neutral(labels_enc_reord) > 0.5  # (batch_size)
        for i in range(batch_size):
            name = filenames_reord[i].decode(sys.getdefaultencoding())
            try:
                crop = inputs_reord[i, ...]  # (input_image_size, input_image_size, 3)
                crop = tools.add_mean_again(crop)
                label_enc = labels_enc_reord[i, :]
                gt_class = int(CommonEncoding.get_gt_class(label_enc))
                gt_coords_enc = CommonEncoding.get_gt_coords(label_enc)
                gt_coords_dec = CommonEncoding.decode_boxes_np(gt_coords_enc, self.opts)
                is_match = mask_match[i]
                is_neutral = mask_neutral[i]
                predicted_class = np.argmax(loc_and_classif[i, 4:])
                if is_neutral:
                    crop_path = os.path.join(batch_dir, name + '_crop' + str(i) + '_NEUTRAL_PRED-' + self.classnames[predicted_class] + '.png')
                else:
                    crop_path = os.path.join(batch_dir, name + '_crop' + str(i) + '_GT-' + self.classnames[gt_class] +
                                             '_PRED-' + self.classnames[predicted_class] + '.png')
                    if gt_class != self.background_id:
                        class_and_coords = np.concatenate([np.expand_dims(gt_class, axis=0), gt_coords_dec], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop = tools.add_bounding_boxes_to_image(crop, class_and_coords, color=(255, 0, 0))
                    else:
                        assert not is_match, 'is_match is True, and gt_class it background_id.'
                    predicted_coords_enc = loc_and_classif[i, :4]
                    predicted_coords_dec = CommonEncoding.decode_boxes_np(predicted_coords_enc, self.opts)
                    if predicted_class != self.background_id:
                        class_and_coords = np.concatenate([np.expand_dims(predicted_class, axis=0), predicted_coords_dec], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop = tools.add_bounding_boxes_to_image(crop, class_and_coords, color=(127, 0, 127))
                cv2.imwrite(crop_path, cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR))

            except:
                print('Error with image ' + name)
                raise

        # Comparisons:
        total_comparisons = (self.opts.n_comparisons_inter + self.opts.n_comparisons_intra) * self.opts.n_images_per_batch
        intra_dir = os.path.join(batch_dir, 'intra_comparisons')
        inter_dir = os.path.join(batch_dir, 'inter_comparisons')
        os.makedirs(intra_dir)
        os.makedirs(inter_dir)
        n_diff = 0
        n_same = 0
        valid_comps = comps_validity_labels[:, 0]  # (total_comparisons)
        comps_labels = comps_validity_labels[:, 1]  # (total_comparisons)
        for i in range(total_comparisons):
            idx1 = comparisons_indices[i, 0]
            idx2 = comparisons_indices[i, 1]
            name1 = filenames_reord[idx1].decode(sys.getdefaultencoding())
            name2 = filenames_reord[idx2].decode(sys.getdefaultencoding())
            try:
                crop1 = inputs_reord[idx1, ...]  # (input_image_size, input_image_size, 3)
                crop2 = inputs_reord[idx2, ...]  # (input_image_size, input_image_size, 3)
                crop1 = tools.add_mean_again(crop1)
                crop2 = tools.add_mean_again(crop2)
                # Box of crop 1:
                label_enc_1 = labels_enc_reord[idx1, :]
                gt_class_1 = int(CommonEncoding.get_gt_class(label_enc_1))
                gt_coords_enc_1 = CommonEncoding.get_gt_coords(label_enc_1)
                gt_coords_dec_1 = CommonEncoding.decode_boxes_np(gt_coords_enc_1, self.opts)
                if gt_class_1 != self.background_id:
                    class_and_coords = np.concatenate([np.expand_dims(gt_class_1, axis=0), gt_coords_dec_1], axis=0)  # (5)
                    class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                    crop1 = tools.add_bounding_boxes_to_image(crop1, class_and_coords, color=(255, 0, 0))
                # Box of crop 2:
                label_enc_2 = labels_enc_reord[idx2, :]
                gt_class_2 = int(CommonEncoding.get_gt_class(label_enc_2))
                gt_coords_enc_2 = CommonEncoding.get_gt_coords(label_enc_2)
                gt_coords_dec_2 = CommonEncoding.decode_boxes_np(gt_coords_enc_2, self.opts)
                if gt_class_2 != self.background_id:
                    class_and_coords = np.concatenate([np.expand_dims(gt_class_2, axis=0), gt_coords_dec_2], axis=0)  # (5)
                    class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                    crop2 = tools.add_bounding_boxes_to_image(crop2, class_and_coords, color=(255, 0, 0))
                # Label and prediction:
                is_valid = valid_comps[i] > 0.5
                if is_valid:
                    gt_comp = 'INVALID'
                else:
                    if comps_labels[i] > 0.5:
                        gt_comp = 'same'
                        n_same += 1
                    else:
                        gt_comp = 'diff'
                        n_diff += 1
                if comparisons_pred[i, 1] > comparisons_pred[i, 0]:
                    pred_comp = 'same'
                else:
                    pred_comp = 'diff'
                mosaic_name = 'comp' + str(i) + '_' + name1 + ' - ' + name2 + '_GT-' + gt_comp + '_PRED-' + pred_comp + '.png'
                if name1 == name2:  # Intra-comparison:
                    mosaic_path = os.path.join(intra_dir, mosaic_name)
                else:  # Inter-comparison
                    mosaic_path = os.path.join(inter_dir, mosaic_name)
                separator = np.zeros(shape=(network.receptive_field_size, 10, 3), dtype=np.float32)
                mosaic = np.concatenate([crop1, separator, crop2], axis=1)  # (2*input_image_size+10, input_image_size, 3)
                cv2.imwrite(mosaic_path, cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2BGR))

            except:
                print('Error with pair ' + name1 + ' - ' + name2)
                raise

        total_valid_comparisons = n_same + n_diff
        proportion_valid_this_batch = float(total_valid_comparisons) / total_comparisons
        if total_valid_comparisons > 0:
            proportion_same_this_batch = float(n_same) / total_valid_comparisons
        else:
            proportion_same_this_batch = 0
        self.proportion_valid_comparisons = (self.proportion_valid_comparisons * (self.batch_count_train_debug - 1) +
                                             proportion_valid_this_batch) / float(self.batch_count_train_debug)
        self.proportion_same_comparisons = (self.proportion_same_comparisons * (self.batch_count_train_debug - 1) +
                                            proportion_same_this_batch) / float(self.batch_count_train_debug)
        print('Percent of valid comparisons. This batch: {:3.2f}. Acummulated: {:3.2f}'.format(proportion_valid_this_batch * 100.0,
                                                                                               self.proportion_valid_comparisons * 100.0))
        print('Percent of same comparisons. This batch: {:3.2f}. Acummulated: {:3.2f}'.format(proportion_same_this_batch * 100.0,
                                                                                               self.proportion_same_comparisons * 100.0))

        return metrics

    def create_folders_for_eval_debug(self):
        self.localizations_dir = os.path.join(self.outdir, 'localizations')
        self.correct_class_dir = os.path.join(self.localizations_dir, 'correct')
        self.wrong_class_dir = os.path.join(self.localizations_dir, 'wrong')
        os.makedirs(self.localizations_dir)
        os.makedirs(self.correct_class_dir)
        os.makedirs(self.wrong_class_dir)
        self.comparisons_dir = os.path.join(self.outdir, 'comparisons')
        self.comparison_same = os.path.join(self.comparisons_dir, 'gt_same')
        self.comparison_diff = os.path.join(self.comparisons_dir, 'gt_diff')
        self.comparison_same_correct = os.path.join(self.comparison_same, 'correct')
        self.comparison_same_wrong = os.path.join(self.comparison_same, 'wrong')
        self.comparison_diff_correct = os.path.join(self.comparison_diff, 'correct')
        self.comparison_diff_wrong = os.path.join(self.comparison_diff, 'wrong')
        os.makedirs(self.comparisons_dir)
        os.makedirs(self.comparison_same)
        os.makedirs(self.comparison_diff)
        os.makedirs(self.comparison_same_correct)
        os.makedirs(self.comparison_same_wrong)
        os.makedirs(self.comparison_diff_correct)
        os.makedirs(self.comparison_diff_wrong)

    def write_eval_debug_info(self, metrics, inputs_reord, labels_enc_reord, loc_and_classif, comparisons_pred, comps_validity_labels, comparisons_indices, filenames):
        # inputs_reord: (batch_size, input_image_size, input_image_size, 3)
        # labels_enc_reord: (batch_size, n_labels)
        # loc_and_classif: (batch_size, 4+nclasses)
        # comparisons_pred: (total_comparisons, 2)
        # comps_validity_labels: (total_comparisons, 2)  [validity, label]
        # comparisons_indices: (total_comparisons, 2)
        # filenames: (n_images_per_batch)
        if self.classnames is None:
            raise Exception('classnames must be specified if debugging.')
        self.batch_count_eval_debug += 1
        batch_size = self.opts.n_crops_per_image * self.opts.n_images_per_batch
        filenames_ext = np.tile(np.expand_dims(filenames, axis=-1), [1, self.opts.n_crops_per_image])  # (n_images_per_batch, n_crops_per_image)
        filenames_reord = np.reshape(filenames_ext, newshape=(batch_size))  # (batch_size)
        mask_match = CommonEncoding.get_mask_match(labels_enc_reord) > 0.5  # (batch_size)
        mask_neutral = CommonEncoding.get_mask_neutral(labels_enc_reord) > 0.5  # (batch_size)

        # Localizations:
        for i in range(batch_size):
            name = filenames_reord[i].decode(sys.getdefaultencoding())
            try:
                crop = inputs_reord[i, ...]  # (input_image_size, input_image_size, 3)
                crop = tools.add_mean_again(crop)
                label_enc = labels_enc_reord[i, :]
                gt_class = int(CommonEncoding.get_gt_class(label_enc))
                is_match = mask_match[i]
                is_neutral = mask_neutral[i]
                predicted_class = np.argmax(loc_and_classif[i, 4:])
                if not is_neutral:
                    file_name = 'batch' + str(self.batch_count_eval_debug) + '_' + name + '_crop' + str(i) + '_GT-' + \
                                self.classnames[gt_class] + '_PRED-' + self.classnames[predicted_class] + '.png'
                    if gt_class == predicted_class:
                        crop_path = os.path.join(self.correct_class_dir, file_name)
                    else:
                        crop_path = os.path.join(self.wrong_class_dir, file_name)

                    if gt_class != self.background_id:
                        gt_coords_enc = CommonEncoding.get_gt_coords(label_enc)
                        gt_coords_dec = CommonEncoding.decode_boxes_np(gt_coords_enc, self.opts)
                        class_and_coords = np.concatenate([np.expand_dims(gt_class, axis=0), gt_coords_dec], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop = tools.add_bounding_boxes_to_image(crop, class_and_coords, color=(255, 0, 0))
                    else:
                        assert not is_match, 'is_match is True, and gt_class it background_id.'
                    if predicted_class != self.background_id:
                        predicted_coords_enc = loc_and_classif[i, :4]
                        predicted_coords_dec = CommonEncoding.decode_boxes_np(predicted_coords_enc, self.opts)
                        class_and_coords = np.concatenate([np.expand_dims(predicted_class, axis=0), predicted_coords_dec], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop = tools.add_bounding_boxes_to_image(crop, class_and_coords, color=(127, 0, 127))
                    cv2.imwrite(crop_path, cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR))

            except:
                print('Error with image ' + name)
                raise

        # Comparisons:
        valid_comps = comps_validity_labels[:, 0]  # (total_comparisons)
        comps_labels = comps_validity_labels[:, 1]  # (total_comparisons)
        total_comparisons = (self.opts.n_comparisons_inter + self.opts.n_comparisons_intra) * self.opts.n_images_per_batch
        n_diff = 0
        n_same = 0
        for i in range(total_comparisons):
            idx1 = comparisons_indices[i, 0]
            idx2 = comparisons_indices[i, 1]
            name1 = filenames_reord[idx1].decode(sys.getdefaultencoding())
            name2 = filenames_reord[idx2].decode(sys.getdefaultencoding())
            is_intra = i < self.opts.n_comparisons_intra * self.opts.n_images_per_batch
            if is_intra:
                print('')
                print('batch ' + str(self.batch_count_eval_debug) + ', comparison ' + str(i))
                print('indices: ' + str(idx1) + ' - ' + str(idx2))
                print('names: ' + name1 + ' - ' + name2)
                is_match_1 = mask_match[idx1] > 0.5
                is_match_2 = mask_match[idx2] > 0.5
                is_neutral_1 = mask_neutral[idx1] > 0.5
                is_neutral_2 = mask_neutral[idx2] > 0.5
                print('match: ' + str(is_match_1) + ' - ' + str(is_match_2))
                print('neutral: ' + str(is_neutral_1) + ' - ' + str(is_neutral_2))
            try:
                is_valid = valid_comps[i] > 0.5
                if is_valid:
                    crop1 = inputs_reord[idx1, ...]  # (input_image_size, input_image_size, 3)
                    crop2 = inputs_reord[idx2, ...]  # (input_image_size, input_image_size, 3)
                    crop1 = tools.add_mean_again(crop1)
                    crop2 = tools.add_mean_again(crop2)
                    # Box of crop 1:
                    label_enc_1 = labels_enc_reord[idx1, :]
                    gt_class_1 = int(CommonEncoding.get_gt_class(label_enc_1))
                    gt_idx_1 = int(CommonEncoding.get_nearest_valid_gt_idx(label_enc_1))
                    if gt_class_1 != self.background_id:
                        gt_coords_enc_1 = CommonEncoding.get_gt_coords(label_enc_1)
                        gt_coords_dec_1 = CommonEncoding.decode_boxes_np(gt_coords_enc_1, self.opts)
                        class_and_coords = np.concatenate([np.expand_dims(gt_class_1, axis=0), gt_coords_dec_1], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop1 = tools.add_bounding_boxes_to_image(crop1, class_and_coords, color=(255, 0, 0))
                    # Box of crop 2:
                    label_enc_2 = labels_enc_reord[idx2, :]
                    gt_class_2 = int(CommonEncoding.get_gt_class(label_enc_2))
                    gt_idx_2 = int(CommonEncoding.get_nearest_valid_gt_idx(label_enc_2))
                    if gt_class_2 != self.background_id:
                        gt_coords_enc_2 = CommonEncoding.get_gt_coords(label_enc_2)
                        gt_coords_dec_2 = CommonEncoding.decode_boxes_np(gt_coords_enc_2, self.opts)
                        class_and_coords = np.concatenate([np.expand_dims(gt_class_2, axis=0), gt_coords_dec_2], axis=0)  # (5)
                        class_and_coords = np.expand_dims(class_and_coords, axis=0)  # (1, 5)
                        crop2 = tools.add_bounding_boxes_to_image(crop2, class_and_coords, color=(255, 0, 0))
                    # Label and prediction:
                    if comps_labels[i] > 0.5:
                        gt_comp = 'same'
                        n_same += 1
                        assert gt_idx_1 == gt_idx_2, 'Boxes marked as same, but associated GT is different.'
                    else:
                        gt_comp = 'diff'
                        n_diff += 1
                        if is_intra and gt_idx_1 == gt_idx_2:
                            raise Exception('Intra comparison, boxes marked as different, but same GT associated.')
                    if comparisons_pred[i, 1] > comparisons_pred[i, 0]:
                        pred_comp = 'same'
                    else:
                        pred_comp = 'diff'
                    if is_intra:
                        print('gt_comp = ' + gt_comp)
                        print('pred_comp = ' + pred_comp)
                        print('gt_class_1 = ' + str(gt_class_1))
                        print('gt_class_2 = ' + str(gt_class_2))
                        print('gt_idx_1 = ' + str(gt_idx_1))
                        print('gt_idx_2 = ' + str(gt_idx_2))
                    # Make the mosaic and save it:
                    mosaic_name = 'batch' + str(self.batch_count_eval_debug) + '_comp' + str(i) + '_' + name1 + ' - ' + name2 + '.png'
                    if gt_comp == pred_comp:
                        if gt_comp == 'same':
                            mosaic_path = os.path.join(self.comparison_same_correct, mosaic_name)
                        else:
                            mosaic_path = os.path.join(self.comparison_diff_correct, mosaic_name)
                    else:
                        if gt_comp == 'same':
                            mosaic_path = os.path.join(self.comparison_same_wrong, mosaic_name)
                        else:
                            mosaic_path = os.path.join(self.comparison_diff_wrong, mosaic_name)
                    if is_intra:
                        print('mosaic_path = ' + mosaic_path)
                    separator = np.zeros(shape=(network.receptive_field_size, 10, 3), dtype=np.float32)
                    mosaic = np.concatenate([crop1, separator, crop2], axis=1)  # (2*input_image_size+10, input_image_size, 3)
                    cv2.imwrite(mosaic_path, cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2BGR))

            except:
                print('Error with pair ' + name1 + ' - ' + name2)
                raise

        total_valid_comparisons = n_same + n_diff
        proportion_valid_this_batch = float(total_valid_comparisons) / total_comparisons
        if total_valid_comparisons > 0:
            proportion_same_this_batch = float(n_same) / total_valid_comparisons
        else:
            proportion_same_this_batch = 0
        self.proportion_valid_comparisons = (self.proportion_valid_comparisons * (self.batch_count_eval_debug - 1) +
                                             proportion_valid_this_batch) / float(self.batch_count_eval_debug)
        self.proportion_same_comparisons = (self.proportion_same_comparisons * (self.batch_count_eval_debug - 1) +
                                            proportion_same_this_batch) / float(self.batch_count_eval_debug)
        print('Percent of valid comparisons. This batch: {:3.2f}. Acummulated: {:3.2f}'.format(proportion_valid_this_batch * 100.0,
                                                                                               self.proportion_valid_comparisons * 100.0))
        print('Percent of same comparisons. This batch: {:3.2f}. Acummulated: {:3.2f}'.format(proportion_same_this_batch * 100.0,
                                                                                               self.proportion_same_comparisons * 100.0))

        return metrics


def classification_loss_and_metric(pred_conf, mask_match, mask_neutral, gt_class, zeros, n_positives):
    # pred_conf: (batch_size, nclasses)
    # mask_match: (batch_size)
    # mask_negative: (batch_size)
    # gt_class: (batch_size)
    # zeros: (batch_size)
    # n_positives: ()
    with tf.variable_scope('conf_loss'):
        gt_class_int = tf.cast(gt_class, tf.int32)
        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_int, logits=pred_conf, name='loss_orig')  # (batch_size)
        loss_positives = tf.where(mask_match, loss_orig, zeros, name='loss_positives')  # (batch_size)
        mask_negatives = tf.logical_and(tf.logical_not(mask_match), tf.logical_not(mask_neutral), name='mask_negatives')  # (batch_size)
        loss_negatives = tf.where(mask_negatives, loss_orig, zeros, name='loss_negatives')
        n_negatives = tf.reduce_sum(tf.cast(mask_negatives, tf.int32), name='n_negatives')  # ()
        loss_pos_scaled = tf.divide(loss_positives, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_pos_scaled')  # (batch_size)
        loss_neg_scaled = tf.divide(loss_negatives, tf.maximum(tf.cast(n_negatives, tf.float32), 1), name='loss_neg_scaled')  # (batch_size)
        loss_conf = tf.reduce_sum(loss_pos_scaled + loss_neg_scaled, name='loss_conf')  # ()

        # Metric:
        predicted_class = tf.argmax(pred_conf, axis=1, output_type=tf.int32)  # (batch_size)
        hits = tf.cast(tf.equal(gt_class_int, predicted_class), tf.float32)  # (batch_size)
        hits_no_neutral = tf.where(tf.logical_not(mask_neutral), hits, zeros)  # (batch_size)
        n_hits = tf.reduce_sum(hits_no_neutral)  # ()
        accuracy_conf = tf.divide(n_hits, tf.maximum(tf.cast(n_negatives + n_positives, tf.float32), 1))  # ()

    return loss_conf, accuracy_conf


def localization_loss_and_metric(pred_coords, mask_match, mask_neutral, gt_coords, zeros, loc_loss_factor, opts):
    # pred_coords: (batch_size, 4)  encoded
    # mask_match: (batch_size)
    # mask_neutral: (batch_size)
    # gt_coords: (batch_size, 4)  encoded
    # zeros: (batch_size)
    with tf.variable_scope('loc_loss'):
        localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, pred_coords)  # (batch_size)
        valids_for_loc = tf.logical_or(mask_match, mask_neutral)  # (batch_size)
        n_valids = tf.reduce_sum(tf.cast(valids_for_loc, tf.int32), name='n_valids_loc')  # ()
        n_valids_safe = tf.maximum(tf.cast(n_valids, tf.float32), 1)
        localization_loss_matches = tf.where(valids_for_loc, localization_loss, zeros, name='loss_match')  # (batch_size)
        loss_loc_summed = tf.reduce_sum(localization_loss_matches, name='loss_summed')  # ()
        loss_loc_scaled = tf.divide(loss_loc_summed, n_valids_safe, name='loss_scaled')  # ()
        loss_loc = tf.multiply(loss_loc_scaled, loc_loss_factor, name='loss_loc')  # ()

        # Metric:
        pred_coords_dec = decode_boxes_tf(pred_coords, opts)  # (batch_size, 4)
        gt_coords_dec = decode_boxes_tf(gt_coords, opts)  # (batch_size, 4)
        iou = compute_iou_tf(pred_coords_dec, gt_coords_dec)  # (batch_size)
        iou_matches = tf.where(valids_for_loc, iou, zeros)  # (batch_size)
        iou_summed = tf.reduce_sum(iou_matches)  # ()
        iou_mean = tf.divide(iou_summed, n_valids_safe, name='iou_mean')  # ()
    return loss_loc, iou_mean


def comparison_loss_and_metric(comparisons_pred, comps_validity_labels, comp_loss_factor):
    # comparisons_pred: (total_comparisons, 2)
    # comps_validity_labels: (total_comparisons, 2)  [validity, label]
    with tf.variable_scope('comp_loss'):
        comparisons_labels_int = tf.cast(comps_validity_labels[:, 1], tf.int32)  # (total_comparisons)
        comparisons_validity = comps_validity_labels[:, 0]  # (total_comparisons)
        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=comparisons_labels_int,
                                                                   logits=comparisons_pred, name='loss_orig')  # (total_comparisons)
        zeros = tf.zeros_like(comparisons_validity, dtype=tf.float32)  # (total_comparisons)
        # TODO: Here we are supressing every case in which there is a neutral crop. If it is an inter-comparison, that's not necessary.
        loss_on_valids = tf.where(comparisons_validity, loss_orig, zeros, name='loss_match')  # (total_comparisons)
        n_valids = tf.reduce_sum(tf.cast(comparisons_validity, tf.float32))  # ()
        n_valids_safe = tf.maximum(n_valids, 1)  # ()
        loss_comp = tf.reduce_sum(loss_on_valids)  # ()
        loss_comp = tf.divide(loss_comp, n_valids_safe, name='loss_summed')
        loss_comp = tf.multiply(loss_comp, comp_loss_factor, name='loss_comp')  # ()

        # Metric:
        predicted_class = tf.argmax(comparisons_pred, axis=1, output_type=tf.int32)  # (total_comparisons)
        hits = tf.cast(tf.equal(comparisons_labels_int, predicted_class), tf.float32)  # (total_comparisons)
        hits_valids = tf.where(comparisons_validity, hits, zeros)  # (total_comparisons)
        n_hits = tf.reduce_sum(hits_valids)  # ()
        accuracy_comp = tf.divide(n_hits, n_valids_safe)  # ()
    return loss_comp, accuracy_comp


def simplify_batch_dimensions(x):
    # x: (n_images_per_batch, n_crops_per_image, ...)
    full_shape = tf.shape(x)
    left_dimensions = full_shape[:2]
    right_dimensions = full_shape[2:]
    batch_size = tf.expand_dims(left_dimensions[0] * left_dimensions[1], axis=0)
    new_shape = tf.concat([batch_size, right_dimensions], axis=0)
    x_reord = tf.reshape(x, shape=new_shape)
    return x_reord


def compute_ar_dc(gt_boxes):
    # gt_boxes (n_gt, 4) [xmin, ymin, width, height]  # Parameterized with the top-left coordinates, and the width and height.
    # Coordinates are relative (between 0 and 1)
    # We consider the square containing the ground truth box.

    gt_boxes_xcenter = gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0
    gt_boxes_ycenter = gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0
    gt_boxes_maxside = np.maximum(gt_boxes[:, 2], gt_boxes[:, 3])
    gt_boxes_square_xmin = np.maximum(gt_boxes_xcenter - gt_boxes_maxside / 2.0, 0)
    gt_boxes_square_ymin = np.maximum(gt_boxes_ycenter - gt_boxes_maxside / 2.0, 0)
    gt_boxes_square_xmax = np.minimum(gt_boxes_xcenter + gt_boxes_maxside / 2.0, 1.0)
    gt_boxes_square_ymax = np.minimum(gt_boxes_ycenter + gt_boxes_maxside / 2.0, 1.0)
    gt_boxes_square = np.stack([gt_boxes_square_xmin, gt_boxes_square_ymin, gt_boxes_square_xmax, gt_boxes_square_ymax], axis=1)  # (n_gt, 4) [xmin, ymin, xmax, ymax]

    # Area ratio:
    area_gt_sq = (gt_boxes_square[:, 2] - gt_boxes_square[:, 0]) * (gt_boxes_square[:, 3] - gt_boxes_square[:, 1])  # (n_gt)
    ar = area_gt_sq  # (n_gt)

    rel_distance_x = 1 - 2 * gt_boxes_xcenter
    rel_distance_y = 1 - 2 * gt_boxes_ycenter

    dc = np.sqrt(np.square(rel_distance_x) + np.square(rel_distance_y))  # (n_gt)
    dc = np.minimum(dc, np.sqrt(2.0))

    return ar, dc


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


def encode_boxes_np(coords_raw, opts):
    # coords_raw: (4)
    xmin = coords_raw[0]
    ymin = coords_raw[1]
    width = coords_raw[2]
    height = coords_raw[3]

    xc = xmin + 0.5 * width
    yc = ymin + 0.5 * height

    dcx = (0.5 - xc) / 0.5
    dcy = (0.5 - yc) / 0.5
    # Between -1 and 1 for the box to lie inside the anchor.

    # Encoding step:
    if opts.encoding_method == 'basic_1':
        dcx_enc = np.tan(dcx * (np.pi / 2.0 - opts.enc_epsilon))
        dcy_enc = np.tan(dcy * (np.pi / 2.0 - opts.enc_epsilon))
        w_enc = CommonEncoding.logit((width - opts.enc_wh_b) / opts.enc_wh_a)
        h_enc = CommonEncoding.logit((height - opts.enc_wh_b) / opts.enc_wh_a)
    elif opts.encoding_method == 'ssd':
        dcx_enc = dcx * 10.0
        dcy_enc = dcy * 10.0
        w_enc = np.log(width) * 5.0
        h_enc = np.log(height) * 5.0
    elif opts.encoding_method == 'no_encode':
        dcx_enc = dcx
        dcy_enc = dcy
        w_enc = width
        h_enc = height
    else:
        raise Exception('Encoding method not recognized.')

    coords_enc = np.stack([dcx_enc, dcy_enc, w_enc, h_enc], axis=0)  # (4)

    return coords_enc  # (4) [dcx_enc, dcy_enc, w_enc, h_enc]

def decode_boxes_tf(coords_enc, opts):
    # coords_enc: (..., 4)
    dcx_enc = coords_enc[..., 0]
    dcy_enc = coords_enc[..., 1]
    w_enc = coords_enc[..., 2]
    h_enc = coords_enc[..., 3]

    # Decoding step:
    if opts.encoding_method == 'basic_1':
        dcx_rel = tf.clip_by_value(tf.atan(dcx_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        dcy_rel = tf.clip_by_value(tf.atan(dcy_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        width = tf.clip_by_value(CommonEncoding.sigmoid(w_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        height = tf.clip_by_value(CommonEncoding.sigmoid(h_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
    elif opts.encoding_method == 'ssd':
        dcx_rel = tf.clip_by_value(dcx_enc * 0.1, -1.0, 1.0)
        dcy_rel = tf.clip_by_value(dcy_enc * 0.1, -1.0, 1.0)
        width = tf.clip_by_value(tf.exp(w_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
        height = tf.clip_by_value(tf.exp(h_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
    elif opts.encoding_method == 'no_encode':
        dcx_rel = tf.clip_by_value(dcx_enc, -1.0, 1.0)
        dcy_rel = tf.clip_by_value(dcy_enc, -1.0, 1.0)
        width = tf.clip_by_value(w_enc, 1.0 / network.receptive_field_size, 1.0)
        height = tf.clip_by_value(h_enc, 1.0 / network.receptive_field_size, 1.0)
    else:
        raise Exception('Encoding method not recognized.')

    xc = 0.5 - dcx_rel * 0.5  # (...)
    yc = 0.5 - dcy_rel * 0.5  # (...)

    xmin = xc - 0.5 * width  # (...)
    ymin = yc - 0.5 * height  # (...)

    coords_raw = tf.stack([xmin, ymin, width, height], axis=-1)  # (..., 4)

    return coords_raw  # (..., 4) [xmin, ymin, width, height]

