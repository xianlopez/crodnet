import tensorflow as tf
import network
import numpy as np
import CommonEncoding


class SingleCellArch:
    def __init__(self, options, nclasses, is_training):
        self.opts = options
        self.nclasses = nclasses + 1 # The last class id is for the background
        self.background_id = self.nclasses - 1
        self.is_training = is_training
        self.n_labels = 7
        self.batch_size = self.opts.n_images_per_batch * self.opts.n_crops_per_image

    def reord_inputs(self, inputs):
        # inputs: (n_images_per_batch, n_crops_per_image, input_image_size, input_image_size, 3)
        inputs_reord = tf.reshape(inputs, shape=(self.batch_size, self.opts.input_image_size, self.opts.input_image_size, 3))
        return inputs_reord

    def reord_labels(self, labels_enc):
        # inputs: (n_images_per_batch, n_crops_per_image, n_labels)
        labels_reord = tf.reshape(labels_enc, shape=(self.batch_size, self.n_labels))
        return labels_reord

    def make(self, inputs, labels_enc, filenames):
        # inputs: (n_images_per_batch, n_crops_per_image, input_image_size, input_image_size, 3)
        # labels: (n_images_per_batch, n_crops_per_image, n_labels)
        inputs_reord = self.reord_inputs(inputs)  # (batch_size, input_image_size, input_image_size, 3)
        labels_enc_reord = self.reord_labels(labels_enc)  # (batch_size, n_labels)
        common_representation = network.common_representation(inputs_reord, self.opts.lcr)  # (batch_size, 1, 1, lcr)
        loc_and_classif = network.localization_and_classification_path(common_representation, self.opts, self.nclasses)  # (batch_size, 1, 1, ?)
        common_representation = tf.squeeze(common_representation, axis=[1, 2])  # (batch_size, lcr)
        loc_and_classif = tf.squeeze(loc_and_classif, axis=[1, 2])  # (batch_size, ?)

    def make_loss(self, common_representation, loc_and_classif, labels_enc_reord):
        pass


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

def decode_boxes(coords_enc, opts, receptive_field_size):
    # coords_enc: (..., 4)
    dcx_enc = coords_enc[..., 0]
    dcy_enc = coords_enc[..., 1]
    w_enc = coords_enc[..., 2]
    h_enc = coords_enc[..., 3]

    # Decoding step:
    if opts.encoding_method == 'basic_1':
        dcx_rel = np.clip(np.arctan(dcx_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        dcy_rel = np.clip(np.arctan(dcy_enc) / (np.pi / 2.0 - opts.enc_epsilon), -1.0, 1.0)
        width = np.clip(CommonEncoding.sigmoid(w_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / receptive_field_size, 1.0)
        height = np.clip(CommonEncoding.sigmoid(h_enc) * opts.enc_wh_a + opts.enc_wh_b, 1.0 / receptive_field_size, 1.0)
    elif opts.encoding_method == 'ssd':
        dcx_rel = np.clip(dcx_enc * 0.1, -1.0, 1.0)
        dcy_rel = np.clip(dcy_enc * 0.1, -1.0, 1.0)
        width = np.clip(np.exp(w_enc * 0.2), 1.0 / receptive_field_size, 1.0)
        height = np.clip(np.exp(h_enc * 0.2), 1.0 / receptive_field_size, 1.0)
    elif opts.encoding_method == 'no_encode':
        dcx_rel = np.clip(dcx_enc, -1.0, 1.0)
        dcy_rel = np.clip(dcy_enc, -1.0, 1.0)
        width = np.clip(w_enc, 1.0 / receptive_field_size, 1.0)
        height = np.clip(h_enc, 1.0 / receptive_field_size, 1.0)
    else:
        raise Exception('Encoding method not recognized.')

    xc = 0.5 - dcx_rel * 0.5  # (...)
    yc = 0.5 - dcy_rel * 0.5  # (...)

    xmin = xc - 0.5 * width  # (...)
    ymin = yc - 0.5 * height  # (...)

    coords_raw = np.stack([xmin, ymin, width, height], axis=-1)  # (..., 4)

    return coords_raw  # (..., 4) [xmin, ymin, width, height]

