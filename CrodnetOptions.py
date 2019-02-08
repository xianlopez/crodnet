class CrodnetOptions:
    def __init__(self):
        # self.step_in_pixels = 16 # 2^(num of max pools)
        self.step_in_pixels = 32 # 2^(num of max pools)
        self.input_image_size = 352
        self.grid_levels_size_pad = [
            (352, 0)
        ]
        self.threshold_ar = 0.05
        self.threshold_pc = 0.99
        self.threshold_dc = 0.5
        self.threshold_ar_neutral = 0.025
        self.threshold_pc_neutral = 0.1
        self.threshold_dc_neutral = 0.75

        self.write_crops = False
        self.write_pc_ar_dc = False

        self.predict_pc = False
        self.predict_coordinates = True
        self.box_per_class = False
        self.compare_boxes = True
        self.pc_loss_factor = 0.1
        self.loc_loss_factor = 1
        self.comp_loss_factor = 0.5

        self.enc_epsilon = 1e-5
        self.enc_wh_a = 2
        self.enc_wh_b = 0.5 * (1 - self.enc_wh_a)

        self.encoding_method = 'ssd'

        self.lcr = 128
        self.min_iou_to_compare = 0.4