

class GridLevel:
    def __init__(self, input_size, pad_abs):
        self.input_size = input_size
        self.pad_abs = pad_abs
        self.input_size_w_pad = self.input_size + 2 * self.pad_abs
        self.pad_rel = float(self.pad_abs) / self.input_size_w_pad
        # Declare attributes that must be defined later:
        self.output_shape = -1
        self.n_boxes = -1
        self.flat_start_pos = -1
        self.rel_box_size = -1

    def set_output_shape(self, output_shape, receptive_field_size):
        self.output_shape = output_shape
        self.n_boxes = self.output_shape * self.output_shape
        self.rel_box_size = float(receptive_field_size) / self.input_size_w_pad

    def set_flat_start_pos(self, flat_start_pos):
        self.flat_start_pos = flat_start_pos