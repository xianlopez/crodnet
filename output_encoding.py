


def get_loc_enc(net_output):
    # net_output: (..., ?)
    return net_output[..., :4]


def get_logits(net_output, predict_pc, predict_dc, predict_cm):
    # net_output: (..., ?)
    n_pos_after_logits = 0
    if predict_pc:
        n_pos_after_logits += 1
    if predict_dc:
        n_pos_after_logits += 1
    if predict_cm:
        n_pos_after_logits += 1
    if n_pos_after_logits == 0:
        return net_output[..., 4:]
    else:
        return net_output[..., 4:-n_pos_after_logits]


def get_pc_enc(net_output, predict_pc, predict_dc, predict_cm):
    # net_output: (..., ?)
    if not predict_pc:
        return None
    else:
        pc_pos = -1
        if predict_dc:
            pc_pos -= 1
        if predict_cm:
            pc_pos -= 1
        return net_output[..., pc_pos]


def get_dc_enc(net_output, predict_pc, predict_dc, predict_cm):
    # net_output: (..., ?)
    if not predict_dc:
        return None
    else:
        dc_pos = -1
        if predict_cm:
            dc_pos -= 1
        return net_output[..., dc_pos]


def get_cm_enc(net_output, predict_pc, predict_dc, predict_cm):
    # net_output: (..., ?)
    if not predict_cm:
        return None
    else:
        return net_output[..., -1]