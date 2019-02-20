


def get_loc_enc(net_output):
    # net_output: (..., ?)
    return net_output[..., :4]


def get_logits(net_output, predict_pc, predict_dc):
    # net_output: (..., ?)
    if predict_pc:
        if predict_dc:
            return net_output[..., 4:-2]
        else:
            return net_output[..., 4:-1]
    elif predict_dc:
        return net_output[..., 4:-1]
    else:
        return net_output[..., 4:]


def get_pc(net_output, predict_pc, predict_dc):
    # net_output: (..., ?)
    if not predict_pc:
        return None
    else:
        if predict_dc:
            return net_output[..., -2]
        else:
            return net_output[..., -1]


def get_dc(net_output, predict_pc, predict_dc):
    # net_output: (..., ?)
    if not predict_dc:
        return None
    else:
        return net_output[..., -1]