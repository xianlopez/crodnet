import tensorflow as tf
import network
import numpy as np
import CommonEncoding
import sys
from BoundingBoxes import BoundingBox, PredictedBox


class SingleCellArch:
    def __init__(self, options, nclasses, is_training):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.is_training = is_training
        self.n_labels = 7
        self.batch_size = self.opts.n_images_per_batch * self.opts.n_crops_per_image

    def make(self, inputs, labels_enc, filenames):
        # inputs: (n_images_per_batch, n_crops_per_image, input_image_size, input_image_size, 3)
        # labels: (n_images_per_batch, n_crops_per_image, n_labels)
        inputs_reord = simplify_batch_dimensions(inputs)  # (batch_size, input_image_size, input_image_size, 3)
        labels_enc_reord = simplify_batch_dimensions(labels_enc)  # (batch_size, n_labels)
        common_representation = network.common_representation(inputs_reord, self.opts.lcr)  # (batch_size, 1, 1, lcr)
        loc_and_classif = network.localization_and_classification_path(common_representation, self.opts, self.nclasses)  # (batch_size, 1, 1, 4+nclasses)
        common_representation = tf.squeeze(common_representation, axis=[1, 2])  # (batch_size, lcr)
        loc_and_classif = tf.squeeze(loc_and_classif, axis=[1, 2])  # (batch_size, 4+nclasses)

    def make_loss(self, common_representation, loc_and_classif, labels_enc_reord):
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

    def classification_loss(self, pred_conf, mask_match, mask_neutral, gt_class, zeros, n_positives):
        # classif_pred: (batch_size, nclasses)
        # mask_match: (batch_size)
        # mask_negative: (batch_size)
        # gt_class: (batch_size)
        # zeros: (batch_size)
        # n_positives: ()
        with tf.variable_scope('classif_loss'):
            loss_orig = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class, logits=pred_conf, name='conf_loss_orig')  # (batch_size)
            loss_positives = tf.where(mask_match, loss_orig, zeros, name='conf_loss_positives')  # (batch_size)
            mask_negatives = tf.logical_and(tf.logical_not(mask_match), tf.logical_not(mask_neutral), name='mask_negatives')  # (batch_size)
            loss_negatives = tf.where(mask_negatives, loss_orig, zeros, name='conf_loss_negatives')
            n_negatives = tf.reduce_sum(tf.cast(mask_negatives, tf.int32), name='n_negatives')  # ()
            loss_pos_scaled = tf.divide(loss_positives, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_pos_scaled')  # (batch_size)
            loss_neg_scaled = tf.divide(loss_negatives, tf.maximum(tf.cast(n_negatives, tf.float32), 1), name='loss_neg_scaled')  # (batch_size)
            loss_conf = tf.reduce_sum(loss_pos_scaled + loss_neg_scaled, name='loss_conf')  # ()
        return loss_conf

    def localization_loss(self, pred_coords, mask_match, gt_coords, zeros, n_positives):
        # pred_coords: (batch_size, 4)
        # mask_match: (batch_size)
        # gt_coords: (batch_size, 4)
        # zeros: (batch_size)
        # n_positives: ()
        localization_loss = CommonEncoding.smooth_L1_loss(gt_coords, pred_coords)  # (batch_size)
        localization_loss_matches = tf.where(mask_match, localization_loss, zeros, name='loc_loss_match')  # (batch_size)
        loss_loc_scaled = tf.divide(localization_loss_matches, tf.maximum(tf.cast(n_positives, tf.float32), 1), name='loss_loc_scaled')  # (batch_size)
        loss_loc = tf.reduce_sum(loss_loc_scaled, name='loss_loc')  # ()
        return loss_loc

    def comparison_loss(self):
        pass

    def encode_gt(self, gt_boxes, filename):
        # Inputs:
        #     gt_boxes: List of GroundTruthBox objects, with relative coordinates (between 0 and 1)
        # Outputs:
        #     encoded_labels: (n_labels)
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
            coordinates_enc = encode_boxes(coordinates, self.opts)  # (4)
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
        coords_raw = decode_boxes(coords_enc, self.opts)  # (4)
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
            pred_coords_raw = decode_boxes(selected_coords, self.opts)  # (4) [xmin, ymin, width, height]
        else:
            # net_output_coords: (4)
            pred_coords_raw = decode_boxes(net_output_coords, self.opts)  # (4) [xmin, ymin, width, height]
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


def encode_boxes(coords_raw, opts):
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

def decode_boxes(coords_enc, opts):
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

