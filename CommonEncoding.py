import tensorflow as tf
import numpy as np


def split_net_output_tf(net_output, opts, nclasses):
    if opts.predict_coordinates:
        if opts.box_per_class:
            if opts.predict_pc:
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl1_pc, cl2...]
                net_output_conf = net_output[..., 0:(nclasses*6):6]
                dcx_enc = net_output[..., 1:(nclasses*6):6]  # (..., n_classes)
                dcy_enc = net_output[..., 2:(nclasses*6):6]  # (..., n_classes)
                w_enc = net_output[..., 3:(nclasses*6):6]  # (..., n_classes)
                h_enc = net_output[..., 4:(nclasses*6):6]  # (..., n_classes)
                net_output_coords = tf.concat([dcx_enc, dcy_enc, w_enc, h_enc], axis=-1)  # (..., 4 * n_classes)
                net_output_pc = net_output[..., 5:(nclasses*6):6]  # (..., n_classes)
            else:
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl2...]
                net_output_conf = net_output[..., 0:(nclasses*5):5]
                dcx_enc = net_output[..., 1:(nclasses*5):5]  # (..., n_classes)
                dcy_enc = net_output[..., 2:(nclasses*5):5]  # (..., n_classes)
                w_enc = net_output[..., 3:(nclasses*5):5]  # (..., n_classes)
                h_enc = net_output[..., 4:(nclasses*5):5]  # (..., n_classes)
                net_output_coords = tf.concat([dcx_enc, dcy_enc, w_enc, h_enc], axis=-1)  # (..., 4 * n_classes)
                net_output_pc = None
        else:
            if opts.predict_pc:
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc, pc]
                net_output_conf = net_output[..., :nclasses]
                net_output_coords = net_output[..., nclasses:(nclasses+4)]  # (..., 4)
                net_output_pc = net_output[..., (nclasses+4)]  # (...)
            else:
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc]
                net_output_conf = net_output[..., :nclasses]
                net_output_coords = net_output[..., nclasses:]  # (..., 4)
                net_output_pc = None
    else:
        # [cl1_conf, cl2_conf, ...]
        net_output_conf = net_output[..., :nclasses]
        net_output_coords = None
        net_output_pc = None
    # net_output_conf: (..., nclasses)
    # net_output_coords: (..., ?)
    # net_output_pc: (..., ?)
    return net_output_conf, net_output_coords, net_output_pc


def put_together_net_output(net_output_conf, net_output_coords, net_output_pc, opts, nclasses):
    # net_output_conf: (..., nclasses)
    # net_output_coords: (..., ?)
    # net_output_pc: (..., ?)
    conf_shape = tf.shape(net_output_conf)
    left_dimensions = conf_shape[:-1]
    if opts.predict_coordinates:
        if opts.box_per_class:
            if opts.predict_pc:
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl1_pc, cl2...]
                concatenated = tf.stack([net_output_conf, net_output_coords[..., 0:(nclasses*4):4],
                                          net_output_coords[..., 1:(nclasses*4):4],
                                          net_output_coords[..., 2:(nclasses*4):4],
                                          net_output_coords[..., 3:(nclasses*4):4],
                                          net_output_pc], axis=-1)  # (..., n_classes, 6)
                net_output = tf.reshape(concatenated, shape=tf.concat([left_dimensions, tf.constant(nclasses*6)], axis=0))
            else:
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl2...]
                concatenated = tf.stack([net_output_conf, net_output_coords[..., 0:(nclasses*4):4],
                                          net_output_coords[..., 1:(nclasses*4):4],
                                          net_output_coords[..., 2:(nclasses*4):4],
                                          net_output_coords[..., 3:(nclasses*4):4]], axis=-1)  # (..., n_classes, 5)
                net_output = tf.reshape(concatenated, shape=tf.concat([left_dimensions, tf.constant(nclasses*5)], axis=0))
        else:
            if opts.predict_pc:
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc, pc]
                net_output = tf.concat([net_output_conf, net_output_coords, net_output_pc], axis=-1)  # (..., n_classes + 5)
            else:
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc]
                net_output = tf.concat([net_output_conf, net_output_coords], axis=-1)  # (..., n_classes + 4)
    else:
        # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl1_pc, cl2...]
        net_output = net_output_conf  # (..., n_classes)
    # net_output: (..., ?)
    return net_output


def get_last_layer_n_channels(opts, nclasses):
    if opts.predict_coordinates:
        if opts.box_per_class:
            if opts.predict_pc:
                n_channels_final = nclasses * 6  # For each class, a confidence, four coordinates and pc.
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl1_pc, cl2...]
            else:
                n_channels_final = nclasses * 5  # For each class, a confidence and four coordinates.
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl2...]
        else:
            if opts.predict_pc:
                n_channels_final = nclasses + 5  # A confidence for each class, plus four coordinates and pc.
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc, pc]
            else:
                n_channels_final = nclasses + 4  # A confidence for each class, plus four coordinates.
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc]
    else:
        n_channels_final = nclasses  # A confidence for each class.
        # [cl1_conf, cl2_conf, ...]
        if opts.predict_pc:
            raise Exception('Option predict_pc not available when not predicting coordinates.')
    return n_channels_final


def split_net_output_np(net_output, opts, nclasses):
    # net_output: (..., ?)
    if opts.predict_coordinates:
        if opts.box_per_class:
            if opts.predict_pc:
                # net_output: (..., 6 * nclasses)
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl1_pc, cl2...]
                net_output_conf = net_output[..., 0:(nclasses*6):6]
                dcx_enc = net_output[..., 1:(nclasses*6):6]  # (..., n_classes)
                dcy_enc = net_output[..., 2:(nclasses*6):6]  # (..., n_classes)
                w_enc = net_output[..., 3:(nclasses*6):6]  # (..., n_classes)
                h_enc = net_output[..., 4:(nclasses*6):6]  # (..., n_classes)
                net_output_coords = np.concatenate([dcx_enc, dcy_enc, w_enc, h_enc], axis=-1)  # (..., 4 * n_classes)
                net_output_pc = net_output[..., 5:(nclasses*6):6]  # (..., n_classes)
            else:
                # net_output: (..., 5 * nclasses)
                # [cl1_conf, cl1_dcx_enc, cl1_dcy_enc, cl1_w_enc, cl1_h_enc, cl2...]
                net_output_conf = net_output[..., 0:(nclasses*5):5]
                dcx_enc = net_output[..., 1:(nclasses*5):5]  # (..., n_classes)
                dcy_enc = net_output[..., 2:(nclasses*5):5]  # (..., n_classes)
                w_enc = net_output[..., 3:(nclasses*5):5]  # (..., n_classes)
                h_enc = net_output[..., 4:(nclasses*5):5]  # (..., n_classes)
                net_output_coords = np.concatenate([dcx_enc, dcy_enc, w_enc, h_enc], axis=-1)  # (..., 4 * n_classes)
                net_output_pc = None
        else:
            if opts.predict_pc:
                # net_output: (..., nclasses + 5)
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc, pc]
                net_output_conf = net_output[..., :nclasses]
                net_output_coords = net_output[..., nclasses:(nclasses+4)]  # (..., 4)
                net_output_pc = net_output[..., (nclasses+4)]  # (...)
            else:
                # net_output: (..., nclasses + 4)
                # [cl1_conf, cl2_conf, ..., dcx_enc, dcy_enc, w_enc, h_enc]
                net_output_conf = net_output[..., :nclasses]
                net_output_coords = net_output[..., nclasses:]  # (..., 4)
                net_output_pc = None
    else:
        # net_output: (..., nclasses)
        # [cl1_conf, cl2_conf, ...]
        net_output_conf = net_output
        net_output_coords = None
        net_output_pc = None
    # net_output_conf: (..., nclasses)
    # net_output_coords: (..., ?)
    # net_output_pc: (..., ?)
    return net_output_conf, net_output_coords, net_output_pc


# This is used in case we compute a set of coordinates per class.
def get_selected_coords_tf(indices_mesh, net_output_coords, nclasses):
    # net_output_coords: (..., 4 * nclasses)
    # [dcx_enc_cl1, dcx_enc_cl2, ..., dcy_enc_cl1, dcy_enc_cl2, ..., w_enc_cl1, w_enc_cl2, ..., h_enc_cl1, h_enc_cl2, ...]
    selected_dcx_enc = tf.gather_nd(net_output_coords[..., :nclasses], indices_mesh)  # (...)
    selected_dcy_enc = tf.gather_nd(net_output_coords[..., nclasses:(2*nclasses)], indices_mesh)  # (...)
    selected_w_enc = tf.gather_nd(net_output_coords[..., (2*nclasses):(3*nclasses)], indices_mesh)  # (...)
    selected_h_enc = tf.gather_nd(net_output_coords[..., (3*nclasses):], indices_mesh)  # (...)
    selected_coords = tf.stack([selected_dcx_enc, selected_dcy_enc, selected_w_enc, selected_h_enc], axis=-1)  # (..., 4)
    return selected_coords


def get_selected_coords_np(net_output_conf, net_output_coords, nclasses, box_per_class):
    # net_output_conf: (..., nclasses)
    # net_output_coords: (..., 4 * nclasses)
    # [dcx_enc_cl1, dcx_enc_cl2, ..., dcy_enc_cl1, dcy_enc_cl2, ..., w_enc_cl1, w_enc_cl2, ..., h_enc_cl1, h_enc_cl2, ...]
    if box_per_class:
        indices = np.argmax(net_output_conf, axis=-1)  # (...)
        indices_exp = np.expand_dims(indices, axis=-1)  # (..., 1)
        selected_dcx_enc = np.take_along_axis(net_output_coords[..., :nclasses], indices_exp, axis=-1)  # (..., 1)
        selected_dcy_enc = np.take_along_axis(net_output_coords[..., nclasses:(2*nclasses)], indices_exp, axis=-1)  # (..., 1)
        selected_w_enc = np.take_along_axis(net_output_coords[..., (2*nclasses):(3*nclasses)], indices_exp, axis=-1)  # (..., 1)
        selected_h_enc = np.take_along_axis(net_output_coords[..., (3*nclasses):], indices_exp, axis=-1)  # (..., 1)
        selected_coords = np.concatenate([selected_dcx_enc, selected_dcy_enc, selected_w_enc, selected_h_enc], axis=-1)  # (..., 4)
        return selected_coords
    else:
        return net_output_coords


def logit(x):
    # Maps (0, 1) to (-inf, +inf)
    return np.log(x / (1.0 - x))


def sigmoid(x):
    # Maps (-inf, +inf) to (0, 1)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_L1_loss(y_true, y_pred, reduce_last_dim=True):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    if reduce_last_dim:
        return tf.reduce_sum(l1_loss, axis=-1)
    else:
        return l1_loss


def get_mask_match(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., 4]


def get_mask_neutral(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., 5]


def get_gt_class(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., 6]


def get_nearest_valid_gt_idx(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., 7]


def get_pc_associated(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., 8]


def get_gt_coords(labels_enc):
    # labels_enc_reord: (..., 9)
    return labels_enc[..., :4]