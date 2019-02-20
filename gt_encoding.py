


def get_mask_match(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 4]


def get_mask_neutral(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 5]


def get_class_id(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 6]


def get_associated_gt_idx(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 7]


def get_pc_enc(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 8]


def get_dc_enc(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., 9]


def get_coords_enc(labels_enc):
    # labels_enc_reord: (..., n_labels)
    return labels_enc[..., :4]