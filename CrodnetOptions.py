class CommonOptions:
    def __init__(self):
        self.n_images_per_batch = 16

        self.predict_pc = False
        self.predict_dc = False
        self.predict_cm = False

        self.lcr = 512  # Length of Common Representation

        self.encoding_method = 'ssd'
        self.enc_epsilon = 1e-5
        self.enc_wh_a = 2
        self.enc_wh_b = 0.5 * (1 - self.enc_wh_a)


class MultiCellOptions(CommonOptions):
    def __init__(self):
        super(MultiCellOptions, self).__init__()
        self.grid_levels_size_pad = [
            (288, 96),
            (352, 64),
            (416, 32)
        ]
        self.debug = False
        self.min_iou_to_compare = 0.4


class SingleCellOptions(CommonOptions):
    def __init__(self):
        super(SingleCellOptions, self).__init__()
        self.n_crops_per_image = 8
        self.n_comparisons_intra = 8
        self.n_comparisons_inter = 1

        self.threshold_ar_low = 0.05
        self.threshold_ar_high = 0.9
        self.threshold_pc = 0.99
        self.threshold_dc = 0.5

        self.threshold_ar_low_neutral = 0.04
        self.threshold_ar_high_neutral = 0.9
        self.threshold_pc_neutral = 0.8
        self.threshold_dc_neutral = 0.7

        self.loc_loss_factor = 1
        self.comp_loss_factor = 1
        self.pc_loss_factor = 0.2
        self.dc_loss_factor = 0.2
        self.cm_loss_factor = 0.5

        self.debug_train = False
        self.debug_eval = False

