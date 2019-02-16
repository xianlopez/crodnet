import cv2
import numpy as np
import Preprocessor
from CommonDataReader import CommonDataReader
import tensorflow as tf


# ======================================================================================================================
class MultiCellDataReader(CommonDataReader):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, opts, multi_cell_arch, split):

        super(MultiCellDataReader, self).__init__(opts, opts.multi_cell_opts.n_images_per_batch)

        self.multi_cell_arch = multi_cell_arch
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.preprocessor = Preprocessor.Preprocessor(self.input_width, self.input_height)

        self.filenames = self.get_filenames(split)
        self.n_images = len(self.filenames)
        self.n_batches = int(self.n_images / self.n_images_per_batch)

        self.reset()

        return

    def reset(self):
        self.remaining_indices = np.arange(self.n_images).tolist()
        self.batch_count = 0

    def build_inputs(self):
        inputs = tf.placeholder(shape=(self.n_images_per_batch, self.input_height, self.input_width, 3), dtype=tf.float32)
        return inputs

    def sample_batch_filenames(self):
        indices = np.random.choice(self.remaining_indices, size=self.n_images_per_batch, replace=False)
        batch_filenames = np.take(self.filenames, indices)  # (n_images_per_batch)
        for idx in indices:
            self.remaining_indices.remove(idx)
        return batch_filenames

    def get_next_batch(self):
        end_of_epoch = False
        self.batch_count += 1
        batch_filenames = self.sample_batch_filenames()  # (n_images_per_batch)
        batch_images = np.zeros(shape=(self.n_images_per_batch, self.input_height, self.input_width, 3), dtype=np.float32)
        batch_bboxes = []
        for img_idx in range(self.n_images_per_batch):
            name = batch_filenames[img_idx]
            image, bboxes = self.read_image_with_bboxes(name)
            # image: (?, ?, 3)
            # bboxes: (n_gt, 7)
            image, bboxes = self.resize_pad_zeros(image, bboxes)
            # image: (input_width, input_height, 3)
            image = self.preprocessor.subtract_mean_np(image)
            batch_images[img_idx, ...] = image
            batch_bboxes.append(bboxes)
        if self.batch_count == self.n_batches:
            assert len(self.remaining_indices) == 0, 'batch_count > n_batches, but remaining_indices is not empty.'
            end_of_epoch = True
            self.reset()
        return batch_images, batch_bboxes, batch_filenames, end_of_epoch

    def resize_pad_zeros(self, image, bboxes):
        height, width, _ = image.shape
        # Resize so it fits totally in the input size:
        scale_width = self.input_width / np.float32(width)
        scale_height = self.input_height / np.float32(height)
        scale = min(scale_width, scale_height)
        new_width = int(np.round(width * scale))
        new_height = int(np.round(height * scale))
        image = cv2.resize(image, (new_width, new_height))
        # Increment on each side:
        increment_height = int(self.input_height - new_height)
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(self.input_width - new_width)
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
        # Warp and shift boxes:
        rel_incr_left = float(increment_left) / new_width
        rel_incr_right = float(increment_right) / new_width
        rel_incr_top = float(increment_top) / new_height
        rel_incr_bottom = float(increment_bottom) / new_height
        for i in range(len(bboxes)):
            bboxes[i][1] = (bboxes[i][1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][2] = (bboxes[i][2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
            bboxes[i][3] = bboxes[i][3] / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][4] = bboxes[i][4] / (1.0 + rel_incr_top + rel_incr_bottom)
        return image, bboxes
