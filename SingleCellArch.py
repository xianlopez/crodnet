import tensorflow as tf
import network
import numpy as np
import CommonEncoding
import sys
from BoundingBoxes import BoundingBox, PredictedBox


class SingleCellArch:
    def __init__(self, options, nclasses):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.n_labels = 7
        self.batch_size = self.opts.n_images_per_batch * self.opts.n_crops_per_image
        self.n_metrics = 3
        self.metric_names = ['accuracy_conf', 'iou_mean', 'accuracy_comp']

    def make(self, inputs, labels_enc, filenames):
        # inputs: (n_images_per_batch, n_crops_per_image, input_image_size, input_image_size, 3)
        # labels: (n_images_per_batch, n_crops_per_image, n_labels)
        inputs_reord = simplify_batch_dimensions(inputs)  # (batch_size, input_image_size, input_image_size, 3)
        labels_enc_reord = simplify_batch_dimensions(labels_enc)  # (batch_size, n_labels)
        common_representation = network.common_representation(inputs_reord, self.opts.lcr)  # (batch_size, 1, 1, lcr)
        loc_and_classif = network.localization_and_classification_path(common_representation, self.opts, self.nclasses)  # (batch_size, 1, 1, 4+nclasses)
        common_representation = tf.squeeze(common_representation, axis=[1, 2])  # (batch_size, lcr)
        loc_and_classif = tf.squeeze(loc_and_classif, axis=[1, 2])  # (batch_size, 4+nclasses)
        comparisons_pred, comparisons_labels, comparisons_indices = self.make_comparisons(common_representation, labels_enc_reord)
        loss, metrics = self.make_loss_and_metrics(loc_and_classif, labels_enc_reord, comparisons_pred,
                                                   comparisons_labels, comparisons_indices)  # ()
        return loc_and_classif, loss, metrics

    def make_comparisons(self, common_representation, labels_enc_reord):
        # common_representation: (batch_size, lcr)
        # labels_enc_reord: (batch_size, n_labels)
        gt_class_id = CommonEncoding.get_gt_class(labels_enc_reord)  # (batch_size)
        nearest_valid_gt_idx = CommonEncoding.get_nearest_valid_gt_idx(labels_enc_reord)  # (batch_size)
        images_range = tf.range(self.opts.n_images_per_batch)  # (n_images_per_batch)

        # Indices of intra comprarisons:
        total_comparisons_intra = self.opts.n_images_per_batch * self.opts.n_comparisons_intra
        image_indices_comp_intra = tf.tile(tf.expand_dims(images_range, axis=-1), [1, self.opts.n_comparisons_intra])  # (n_images_per_batch, n_comparisons_intra)
        image_indices_comp_intra = tf.reshape(image_indices_comp_intra, (total_comparisons_intra))  # (total_comparisons_intra)
        # image_indices_comp_intra: [0, 0, ..., 0, 1, 1, ..., 1, ..., n_images_per_batch, n_images_per_batch, ..., n_images_per_batch]
        image_indices_comp_intra_exp = tf.tile(tf.expand_dims(image_indices_comp_intra, axis=-1), [1, 2])  # (total_comparisons_intra, 2)
        # Generate random indices relative to their image:
        crop_indices_intra = tf.random.uniform(shape=(total_comparisons_intra, 2), maxval=self.opts.n_crops_per_image, dtype=tf.int32)
        # Convert this relative indices to absolute:
        random_indices_intra = crop_indices_intra + self.opts.n_crops_per_image * image_indices_comp_intra_exp  # (total_comparisons_intra, 2)

        # Indices of inter comparisons:
        total_comparisons_inter = self.opts.n_images_per_batch * self.opts.n_comparisons_inter
        crop_indices_inter = tf.random.uniform(shape=(total_comparisons_inter, 2), maxval=self.opts.n_crops_per_image, dtype=tf.int32)
        image_indices_comp_inter_left = tf.random.uniform(shape=(total_comparisons_inter), maxval=self.opts.n_images_per_batch, dtype=tf.int32)
        image_indices_distance = tf.random.uniform(shape=(total_comparisons_inter), minval=1, maxval=self.opts.n_images_per_batch, dtype=tf.int32)
        image_indices_comp_inter_right = tf.floormod(image_indices_comp_inter_left + image_indices_distance, self.opts.n_images_per_batch)
        image_indices_comp_inter = tf.stack([image_indices_comp_inter_left, image_indices_comp_inter_right], axis=-1)  # (total_comparisons_inter, 2)
        random_indices_inter = crop_indices_inter + self.opts.n_crops_per_image * image_indices_comp_inter  # (total_comparisons_inter, 2)

        # CRs of comparisons:
        indices_all_comparisons = tf.concat([random_indices_intra, random_indices_inter], axis=0)  # (total_comparisons, 2)
        crs_all_comparisons = tf.gather(common_representation, indices_all_comparisons, axis=0)  # (total_comparisons, 2, lcr)
        print('crs_all_comparisons.shape = ' + str(crs_all_comparisons.shape))

        # Comparisons:
        comparisons_pred = network.comparison(crs_all_comparisons, self.opts.lcr)  # (total_comparisons, 2)
        print('comparisons_pred.shape = ' + str(comparisons_pred.shape))

        # Labels of comparisons:
        gt_idx_intra_comp = tf.gather(nearest_valid_gt_idx, random_indices_intra, axis=0)  # (total_comparisons_intra, 2)
        print('gt_idx_intra_comp.shape = ' + str(gt_idx_intra_comp.shape))
        same_gt_idx = tf.less(tf.abs(gt_idx_intra_comp[:, 0] - gt_idx_intra_comp[:, 1]), 0.5)  # (total_comparisons_intra)
        gt_class_intra_comp = tf.gather(gt_class_id, random_indices_intra, axis=0)  # (total_comparisons_intra, 2)
        any_background = tf.less(tf.abs(gt_class_intra_comp - self.background_id), 0.5)  # (total_comparisons_intra, 2)
        any_background = tf.reduce_any(any_background, axis=1)  # (total_comparisons_intra)
        labels_comp_intra = tf.logical_and(same_gt_idx, tf.logical_not(any_background))  # (total_comparisons_intra)
        labels_comp_inter = tf.zeros(shape=(total_comparisons_inter), dtype=tf.bool)
        labels_all_comparisons = tf.concat([labels_comp_intra, labels_comp_inter], axis=0)  # (total_comparisons)

        return comparisons_pred, labels_all_comparisons, indices_all_comparisons

    def make_loss_and_metrics(self, loc_and_classif, labels_enc_reord, comparisons_pred, comparisons_labels, comparisons_indices):
        # common_representation: (batch_size, lcr)
        # loc_and_classif: (batch_size, 4+nclasses)
        # labels_enc_reord: (batch_size, n_labels)

        pred_conf = loc_and_classif[:, 4:]  # (batch_size, nclasses)
        pred_coords = loc_and_classif[:, :4]  # (batch_size, 4)

        mask_match = CommonEncoding.get_mask_match(labels_enc_reord)  # (batch_size)
        mask_neutral = CommonEncoding.get_mask_neutral(labels_enc_reord)  # (batch_size)
        gt_class_ids = CommonEncoding.get_gt_class(labels_enc_reord)  # (batch_size)
        gt_coords = CommonEncoding.get_gt_coords(labels_enc_reord)  # (batch_size)

        zeros = tf.zeros_like(mask_match)  # (batch_size)
        n_positives = tf.reduce_sum(tf.cast(mask_match, tf.int32), name='n_positives')  # ()

        conf_loss, accuracy_conf = classification_loss_and_metric(pred_conf, mask_match, mask_neutral, gt_class_ids, zeros, n_positives)
        tf.summary.scalar("conf_loss", conf_loss)
        tf.summary.scalar("accuracy_conf", accuracy_conf)
        total_loss = conf_loss

        loc_loss, iou_mean = localization_loss_and_metric(pred_coords, mask_match, gt_coords, zeros, n_positives, self.opts.loc_loss_factor, self.opts)
        tf.summary.scalar("loc_loss", loc_loss)
        tf.summary.scalar("iou_mean", iou_mean)
        total_loss += loc_loss

        comp_loss, accuracy_comp = comparison_loss_and_metric(comparisons_pred, comparisons_labels, comparisons_indices, mask_match, self.opts.comp_loss_factor)
        tf.summary.scalar("comp_loss", comp_loss)
        tf.summary.scalar("accuracy_comp", accuracy_comp)
        total_loss += comp_loss

        metrics = tf.stack([accuracy_conf, iou_mean, accuracy_comp])  # (n_metrics)

        return total_loss, metrics

    def get_input_shape(self):
        input_shape = [network.receptive_field_size, network.receptive_field_size]
        return input_shape

    def encode_gt(self, gt_boxes):
        # Inputs:
        #     gt_boxes: List of GroundTruthBox objects, with relative coordinates (between 0 and 1)
        # Outputs:
        #     encoded_labels: (n_labels)
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

            ar, dc = compute_ar_dc(gt_vec)  # All variables have shape (n_gt)
            pc = gt_pc_incoming  # (n_gt)

            # Positive boxes:
            mask_ar = ar > self.opts.threshold_ar  # (n_gt)
            mask_pc = pc > self.opts.threshold_pc  # (n_gt)
            mask_dc = dc < self.opts.threshold_dc  # (n_gt)
            mask_thresholds = mask_ar & mask_pc & mask_dc  # (n_gt)
            any_match = np.any(mask_thresholds)  # ()

            dc_masked = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt), dtype=np.float32) ) # (n_gt)

            nearest_valid_gt_idx = np.argmin(dc_masked)  # ()

            # Neutral boxes:
            mask_ar_neutral = ar > self.opts.threshold_ar_neutral  # (n_gt)
            mask_pc_neutral = pc > self.opts.threshold_pc_neutral  # (n_gt)
            mask_dc_neutral = dc < self.opts.threshold_dc_neutral  # (n_gt)
            mask_thresholds_neutral = mask_ar_neutral & mask_pc_neutral & mask_dc_neutral  # (n_gt)
            any_neutral = np.any(mask_thresholds_neutral, axis=0)  # ()
            is_neutral = np.logical_and(any_neutral, np.logical_not(any_match))  # ()

            # Get the coordinates and the class id of the gt box matched:
            coordinates = gt_vec[nearest_valid_gt_idx, :]  # (4)
            coordinates_enc = encode_boxes_np(coordinates, self.opts)  # (4)
            class_id = gt_class_ids[nearest_valid_gt_idx]

            # Negative boxes:
            is_negative = np.logical_and(np.logical_not(any_match), np.logical_not(is_neutral))  # ()
            if is_negative:
                class_id = self.background_id

            # Percent contained associated with each anchor box.
            # This is the PC of the assigned gt box, if there is any, or otherwise the maximum PC it has.
            if any_match:
                pc_associated = pc[nearest_valid_gt_idx]
            else:
                pc_associated = np.max(pc)

            # Put all together in one array:
            labels_enc = np.stack([any_match.astype(np.float32),
                                   is_neutral.astype(np.float32),
                                   class_id.astype(np.float32),
                                   nearest_valid_gt_idx.astype(np.float32),
                                   pc_associated])
            labels_enc = np.concatenate([coordinates_enc, labels_enc])  # (9)

        else:
            labels_enc = np.zeros(shape=(9), dtype=np.float32)
            labels_enc[6] = self.background_id

        return labels_enc

    def encode_gt_from_array(self, gt_boxes):
        # gt_boxes: (n_gt, 6) [class_id, xmin, ymin, width, height, pc]
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

            dc_masked = np.where(mask_thresholds, dc, np.infty * np.ones(shape=(n_gt), dtype=np.float32) ) # (n_gt)

            nearest_valid_gt_idx = np.argmin(dc_masked)  # ()

            # Neutral boxes:
            mask_ar_neutral = ar > self.opts.threshold_ar_neutral  # (n_gt)
            mask_pc_neutral = pc > self.opts.threshold_pc_neutral  # (n_gt)
            mask_dc_neutral = dc < self.opts.threshold_dc_neutral  # (n_gt)
            mask_thresholds_neutral = mask_ar_neutral & mask_pc_neutral & mask_dc_neutral  # (n_gt)
            any_neutral = np.any(mask_thresholds_neutral, axis=0)  # ()
            is_neutral = np.logical_and(any_neutral, np.logical_not(any_match))  # ()

            # Get the coordinates and the class id of the gt box matched:
            coordinates = gt_coords[nearest_valid_gt_idx, :]  # (4)
            coordinates_enc = encode_boxes_np(coordinates, self.opts)  # (4)
            class_id = gt_class_ids[nearest_valid_gt_idx]

            # Negative boxes:
            is_negative = np.logical_and(np.logical_not(any_match), np.logical_not(is_neutral))  # ()
            if is_negative:
                class_id = self.background_id

            # Percent contained associated with each anchor box.
            # This is the PC of the assigned gt box, if there is any, or otherwise the maximum PC it has.
            if any_match:
                pc_associated = pc[nearest_valid_gt_idx]
            else:
                pc_associated = np.max(pc)

            # Put all together in one array:
            labels_enc = np.stack([any_match.astype(np.float32),
                                   is_neutral.astype(np.float32),
                                   class_id.astype(np.float32),
                                   nearest_valid_gt_idx.astype(np.float32),
                                   pc_associated])
            labels_enc = np.concatenate([coordinates_enc, labels_enc])  # (9)

        else:
            labels_enc = np.zeros(shape=(9), dtype=np.float32)
            labels_enc[6] = self.background_id

        return labels_enc  # (9)

    def decode_gt(self, labels_enc, remove_duplicated=True):
        # Inputs:
        #     labels_enc: numpy array of dimension (9).
        #     remove_duplicated: Many default boxes can be assigned to the same ground truth box. Therefore,
        #                        when decoding, we can find several times the same box. If we set this to True, then
        #                        the dubplicated boxes are deleted.
        # Outputs:
        #     gt_boxes: List of BoundingBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.
        gt_boxes = []
        coords_enc = labels_enc[:4]  # (4)
        coords_raw = decode_boxes_np(coords_enc, self.opts)  # (4)
        if labels_enc[4] > 0.5:
            classid = int(np.round(labels_enc[6]))
            percent_contained = np.clip(labels_enc[8], 0.0, 1.0)
            gtbox = BoundingBox(coords_raw, classid, percent_contained)
            gt_boxes.append(gtbox)
        return gt_boxes

    def decode_preds(self, net_output_nobatch, th_conf):
        # Inputs:
        #     net_output_nobatch: numpy array of dimension nboxes x 9.
        # Outputs:
        #     predictions: List of PredictedBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.

        net_output_conf, net_output_coords, net_output_pc = CommonEncoding.split_net_output_np(net_output_nobatch, self.opts, self.nclasses)
        # net_output_conf: (nclasses)
        # net_output_coords: (?)
        # net_output_pc: (?)
        if self.opts.box_per_class:
            # net_output_coords: (4 * nclasses)
            selected_coords = CommonEncoding.get_selected_coords_np(net_output_conf, net_output_coords, self.nclasses, self.opts.box_per_class)  # (4)
            pred_coords_raw = decode_boxes_np(selected_coords, self.opts)  # (4) [xmin, ymin, width, height]
        else:
            # net_output_coords: (4)
            pred_coords_raw = decode_boxes_np(net_output_coords, self.opts)  # (4) [xmin, ymin, width, height]
        class_id = np.argmax(net_output_conf)  # ()
        predictions = []
        if class_id != self.background_id:
            gtbox = PredictedBox(pred_coords_raw, class_id, net_output_conf[class_id])
            if self.opts.predict_pc:
                if self.opts.box_per_class:
                    # net_output_pc: (nboxes, nclasses)
                    gtbox.pc = net_output_pc[class_id]
                else:
                    # net_output_pc: (nboxes)
                    gtbox.pc = net_output_pc
            predictions.append(gtbox)
        return predictions


def classification_loss_and_metric(pred_conf, mask_match, mask_neutral, gt_class, zeros, n_positives):
    # pred_conf: (batch_size, nclasses)
    # mask_match: (batch_size)
    # mask_negative: (batch_size)
    # gt_class: (batch_size)
    # zeros: (batch_size)
    # n_positives: ()
    with tf.variable_scope('conf_loss'):
        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class, logits=pred_conf, name='loss_orig')  # (batch_size)
        loss_positives = tf.where(mask_match, loss_orig, zeros, name='loss_positives')  # (batch_size)
        mask_negatives = tf.logical_and(tf.logical_not(mask_match), tf.logical_not(mask_neutral), name='mask_negatives')  # (batch_size)
        loss_negatives = tf.where(mask_negatives, loss_orig, zeros, name='loss_negatives')
        n_negatives = tf.reduce_sum(tf.cast(mask_negatives, tf.int32), name='n_negatives')  # ()
        loss_pos_scaled = tf.divide(loss_positives, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_pos_scaled')  # (batch_size)
        loss_neg_scaled = tf.divide(loss_negatives, tf.maximum(tf.cast(n_negatives, tf.float32), 1), name='loss_neg_scaled')  # (batch_size)
        loss_conf = tf.reduce_sum(loss_pos_scaled + loss_neg_scaled, name='loss_conf')  # ()

        # Metric:
        predicted_class = tf.argmax(pred_conf, axis=1)  # (batch_size)
        hits = tf.equal(gt_class, predicted_class)  # (batch_size)
        hits_no_neutral = tf.where(tf.logical_not(mask_neutral), hits, zeros)  # (batch_size)
        n_hits = tf.reduce_sum(hits_no_neutral)  # ()
        accuracy_conf = tf.divide(n_hits, tf.maximum(tf.cast(n_negatives + n_positives, tf.float32), 1))  # ()
    return loss_conf, accuracy_conf


def localization_loss_and_metric(pred_coords, mask_match, gt_coords, zeros, n_positives, loc_loss_factor, opts):
    # pred_coords: (batch_size, 4)  encoded
    # mask_match: (batch_size)
    # gt_coords: (batch_size, 4)  encoded
    # zeros: (batch_size)
    # n_positives: ()
    with tf.variable_scope('loc_loss'):
        n_positives_safe = tf.maximum(tf.cast(n_positives, tf.float32), 1)
        localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, pred_coords)  # (batch_size)
        localization_loss_matches = tf.where(mask_match, localization_loss, zeros, name='loss_match')  # (batch_size)
        loss_loc_scaled = tf.divide(localization_loss_matches, n_positives_safe, name='loss_scaled')  # (batch_size)
        loss_loc = tf.reduce_sum(loss_loc_scaled, name='loss_summed')  # ()
        loss_loc = tf.multiply(loss_loc, loc_loss_factor, name='loss_loc')  # ()

        # Metric:
        pred_coords_dec = decode_boxes_tf(pred_coords, opts)  # (batch_size, 4)
        gt_coords_dec = decode_boxes_tf(gt_coords, opts)  # (batch_size, 4)
        iou = compute_iou_tf(pred_coords_dec, gt_coords_dec)  # (batch_size)
        iou_matches = tf.where(mask_match, iou, zeros)  # (batch_size)
        iou_mean = tf.divide(iou_matches, n_positives_safe, name='iou_mean')  # ()
    return loss_loc, iou_mean


def comparison_loss_and_metric(comparisons_pred, comparisons_labels, comparisons_indices, mask_match, comp_loss_factor):
    # comparisons_pred: (total_comparisons, 2)
    # comparisons_labels: (total_comparisons)
    # comparisons_indices: (total_comparisons, 2)
    # mask_match: (batch_size)
    with tf.variable_scope('comp_loss'):
        comparisons_labels_int = tf.cast(comparisons_labels, tf.int32)
        loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=comparisons_labels_int,
                                                                   logits=comparisons_pred, name='loss_orig')  # (total_comparisons)
        comparisons_matches = tf.gather(mask_match, comparisons_indices, axis=0)  # (total_comparisons, 2)
        print('comparisons_matches.shape = ' + str(comparisons_matches.shape))
        any_match = tf.reduce_any(comparisons_matches, axis=1)  # (total_comparisons)
        zeros = tf.zeros_like(any_match, dtype=tf.float32)  # (total_comparisons)
        loss_matches = tf.where(any_match, loss_orig, zeros, name='loss_match')  # (total_comparisons)
        n_matches = tf.reduce_sum(tf.cast(any_match, tf.int32))  # ()
        n_matches_safe = tf.maximum(n_matches, 1)  # ()
        loss_comp = tf.reduce_sum(loss_matches)  # ()
        loss_comp = tf.divide(loss_comp, n_matches_safe, name='loss_summed')
        loss_comp = tf.multiply(loss_comp, comp_loss_factor, name='loss_comp')  # ()

        # Metric:
        predicted_class = tf.argmax(comparisons_pred, axis=1)  # (total_comparisons)
        hits = tf.equal(comparisons_labels_int, predicted_class)  # (total_comparisons)
        hits_matches = tf.where(any_match, hits, zeros)  # (total_comparisons)
        n_hits = tf.reduce_sum(hits_matches)  # ()
        accuracy_comp = tf.divide(n_hits, n_matches_safe)  # ()
    return loss_comp, accuracy_comp


def simplify_batch_dimensions(x):
    # x: (n_images_per_batch, n_crops_per_image, ...)
    full_shape = tf.shape(x)
    left_dimensions = full_shape[:2]
    right_dimensions = full_shape[2:]
    batch_size = left_dimensions[0] * left_dimensions[1]
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

def decode_boxes_np(coords_enc, opts):
    # coords_enc: (..., 4)
    dcx_enc = coords_enc[..., 0]
    dcy_enc = coords_enc[..., 1]
    w_enc = coords_enc[..., 2]
    h_enc = coords_enc[..., 3]

    # Decoding step:
    if opts.encoding_method == 'basic_1':
        dcx_rel = np.clip(np.arctan(dcx_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        dcy_rel = np.clip(np.arctan(dcy_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        width = np.clip(CommonEncoding.sigmoid(w_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        height = np.clip(CommonEncoding.sigmoid(h_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
    elif opts.encoding_method == 'ssd':
        dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)
        dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)
        width = np.clip(np.exp(w_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
        height = np.clip(np.exp(h_enc * 0.2), 1.0 / network.receptive_field_size, 1.0)
    elif opts.encoding_method == 'no_encode':
        dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
        dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
        width = np.clip(w_enc, 1.0 / network.receptive_field_size, 1.0)
        height = np.clip(h_enc, 1.0 / network.receptive_field_size, 1.0)
    else:
        raise Exception('Encoding method not recognized.')

    xc = 0.5 - dcx_rel * 0.5  # (...)
    yc = 0.5 - dcy_rel * 0.5  # (...)

    xmin = xc - 0.5 * width  # (...)
    ymin = yc - 0.5 * height  # (...)

    coords_raw = np.stack([xmin, ymin, width, height], axis=-1)  # (..., 4)

    return coords_raw  # (..., 4) [xmin, ymin, width, height]

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

