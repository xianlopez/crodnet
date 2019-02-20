import tensorflow as tf
import numpy as np
import network


def encode_boxes_wrt_anchor_np(coords_raw, opts):
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
        w_enc = logit((width - opts.enc_wh_b) / opts.enc_wh_a)
        h_enc = logit((height - opts.enc_wh_b) / opts.enc_wh_a)
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


def decoding_split_np(dcx_enc, dcy_enc, w_enc, h_enc, opts):
    if opts.encoding_method == 'basic_1':
        dcx_rel = np.clip(np.arctan(dcx_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        dcy_rel = np.clip(np.arctan(dcy_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        width = np.clip(sigmoid(w_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        height = np.clip(sigmoid(h_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
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
    return dcx_rel, dcy_rel, width, height


def decoding_split_tf(dcx_enc, dcy_enc, w_enc, h_enc, opts):
    if opts.encoding_method == 'basic_1':
        dcx_rel = tf.clip_by_value(tf.atan(dcx_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        dcy_rel = tf.clip_by_value(tf.atan(dcy_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        width = tf.clip_by_value(sigmoid(w_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
        height = tf.clip_by_value(sigmoid(h_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / network.receptive_field_size, 1.0)
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
    return dcx_rel, dcy_rel, width, height


def decode_boxes_wrt_anchor_tf(coords_enc, opts):
    # coords_enc: (..., 4)
    dcx_enc = coords_enc[..., 0]
    dcy_enc = coords_enc[..., 1]
    w_enc = coords_enc[..., 2]
    h_enc = coords_enc[..., 3]

    # Decoding step:
    dcx_rel, dcy_rel, width, height = decoding_split_tf(dcx_enc, dcy_enc, w_enc, h_enc, opts)

    xc = 0.5 - dcx_rel * 0.5  # (...)
    yc = 0.5 - dcy_rel * 0.5  # (...)
    xmin = xc - 0.5 * width  # (...)
    ymin = yc - 0.5 * height  # (...)
    coords_raw = tf.stack([xmin, ymin, width, height], axis=-1)  # (..., 4)

    return coords_raw  # (..., 4) [xmin, ymin, width, height]


def decode_boxes_wrt_anchor_np(coords_enc, opts):
    # coords_enc: (..., 4)
    dcx_enc = coords_enc[..., 0]
    dcy_enc = coords_enc[..., 1]
    w_enc = coords_enc[..., 2]
    h_enc = coords_enc[..., 3]

    # Decoding step:
    dcx_rel, dcy_rel, width, height = decoding_split_np(dcx_enc, dcy_enc, w_enc, h_enc, opts)

    xc = 0.5 - dcx_rel * 0.5  # (...)
    yc = 0.5 - dcy_rel * 0.5  # (...)
    xmin = xc - 0.5 * width  # (...)
    ymin = yc - 0.5 * height  # (...)
    coords_raw = np.stack([xmin, ymin, width, height], axis=-1)  # (..., 4)

    return coords_raw  # (..., 4) [xmin, ymin, width, height]


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


def decode_pc_tf(pc_enc):
    return decode_pc_or_dc_tf(pc_enc)


def decode_dc_tf(dc_enc):
    return decode_pc_or_dc_tf(dc_enc)


def decode_pc_np(pc_enc):
    return decode_pc_or_dc_np(pc_enc)


def decode_dc_np(dc_enc):
    return decode_pc_or_dc_np(dc_enc)


def decode_pc_or_dc_tf(x_enc):
    # x_enc: any shape
    x_dec = tf.clip_by_value(x_enc * 0.1, 0.0, 1.0)
    return x_dec


def decode_pc_or_dc_np(x_enc):
    # x_enc: any shape
    x_dec = np.clip(x_enc * 0.1, 0.0, 1.0)
    return x_dec


def encode_pc_or_dc_np(x_dec):
    # x_dec: any shape
    x_enc = x_dec * 10
    return x_enc











