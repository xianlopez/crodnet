import tensorflow as tf
import tools
import logging
import os
import cv2
import numpy as np
import Preprocessor
import sys
import Resizer
from BoundingBoxes import BoundingBox


# ======================================================================================================================
class MultiCellDataReader:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, opts, batch_size, multi_cell_arch):
        self.multi_cell_arch = multi_cell_arch
        self.batch_size = batch_size
        self.opts = opts
        input_width = input_shape[0]
        input_height = input_shape[1]
        self.preprocessor = Preprocessor.Preprocessor(input_width, input_height)
        self.resize_function = Resizer.ResizerWithLabels(input_width, input_height).get_resize_func(self.opts.resize_method)
        self.nimages = None
        self.init_op = None
        self.dirdata = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name)
        self.img_extension, self.classnames = tools.process_dataset_config(os.path.join(self.dirdata, 'dataset_info.xml'))
        return

    # ------------------------------------------------------------------------------------------------------------------
    def build_iterator(self):
        batched_dataset, self.nimages = self.build_batched_dataset()
        print('Number of images: ' + str(self.nimages))
        iterator = tf.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
        inputs, labels, filenames = iterator.get_next(name='iterator-output')
        self.init_op = iterator.make_initializer(batched_dataset, name='train_init_op')
        return inputs, labels, filenames

    # ------------------------------------------------------------------------------------------------------------------
    def build_batched_dataset(self):

        filenames = self.get_filenames()
        batched_dataset = self.build_dataset(filenames)

        return batched_dataset, len(filenames)

    # ------------------------------------------------------------------------------------------------------------------
    def get_filenames(self, split):
        if split != 'train' and split != 'val':
            raise Exception('Split name not recognized.')
        list_file = os.path.join(self.dirdata, split + '_files.txt')
        try:
            with open(list_file, 'r') as fid:
                filenamesnoext = fid.read().splitlines()
            for i in range(len(filenamesnoext)):
                filenamesnoext[i] = tools.adapt_path_to_current_os(filenamesnoext[i])
        except FileNotFoundError as ex:
            logging.error('File ' + list_file + ' does not exist.')
            logging.error(str(ex))
            raise
        # Remove data or shuffle:
        if self.opts.percent_of_data != 100:
            # Remove data:
            indexes = np.random.choice(np.arange(len(filenamesnoext)),
                                       int(self.opts.percent_of_data / 100.0 * len(filenamesnoext)), replace=False)
        else:
            # Shuffle data at least:
            indexes = np.arange(len(filenamesnoext))
            if self.opts.shuffle_data:
                np.random.shuffle(indexes)
        aux = filenamesnoext
        filenamesnoext = []
        for i in range(len(indexes)):
            filenamesnoext.append(aux[indexes[i]])
        # Remove the remaining examples that do not fit in a batch.
        if len(filenamesnoext) % self.opts.n_images_per_batch != 0:
            aux = filenamesnoext
            filenamesnoext = []
            for i in range(len(aux) - (len(aux) % self.opts.n_images_per_batch)):
                filenamesnoext.append(aux[i])
        assert len(filenamesnoext) % self.opts.n_images_per_batch == 0, 'Number of images is not a multiple of n_images_per_batch'
        return filenamesnoext

    # ------------------------------------------------------------------------------------------------------------------
    def build_dataset(self, filenames):
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_func, num_parallel_calls=self.opts.num_workers)  # image, bboxes, filename
        dataset = dataset.map(self.resize_func_extended, num_parallel_calls=self.opts.num_workers)  # image, bboxes, filename
        dataset = dataset.map(self.preprocess_func, num_parallel_calls=self.opts.num_workers)  # image, bboxes, filename
        dataset = dataset.map(self.encode_boxes, num_parallel_calls=self.opts.num_workers)  # image, label_enc, filename
        return dataset.batch(self.batch_size)

    # ------------------------------------------------------------------------------------------------------------------
    def encode_boxes(self, image, bboxes, filename):
        (label_enc) = tf.py_func(self.encode_boxes_np, [bboxes], (tf.float32))
        label_enc.set_shape((self.multi_cell_arch.n_boxes, self.multi_cell_arch.n_labels))
        return image, label_enc, filename

    # ------------------------------------------------------------------------------------------------------------------
    def resize_func_extended(self, image, bboxes, filename):
        image, bboxes = self.resize_function(image, bboxes)
        return image, bboxes, filename

    # ------------------------------------------------------------------------------------------------------------------
    def parse_func(self, filename):
        (image, bboxes) = tf.py_func(self.read_image_with_bboxes, [filename], (tf.float32, tf.float32))
        bboxes.set_shape((None, 6))  # (nboxes, [class_id, x_min, y_min, width, height, pc])
        image.set_shape((None, None, 3))  # (height, width, channels)
        return image, bboxes, filename

    # ------------------------------------------------------------------------------------------------------------------
    def preprocess_func(self, image, label, filename):
        image = self.preprocessor.subtract_mean(image)
        return image, label, filename

    # ------------------------------------------------------------------------------------------------------------------
    def encode_boxes_np(self, boxes_array):
        nboxes = boxes_array.shape[0]
        bboxes = []
        for i in range(nboxes):
            class_id = int(np.round(boxes_array[i, 0]))
            bboxes.append(BoundingBox([boxes_array[i, 1], boxes_array[i, 2],
                                       boxes_array[i, 3], boxes_array[i, 4]], class_id))
        encoded_label = self.multi_cell_arch.encode_gt(bboxes)
        return encoded_label

    # ------------------------------------------------------------------------------------------------------------------
    def read_image_with_bboxes(self, filename):
        dirimg = os.path.join(self.dirdata, "images")
        dirann = os.path.join(self.dirdata, "annotations")
        filename = filename.decode(sys.getdefaultencoding())
        try:
            imagefile = os.path.join(dirimg, filename + self.img_extension)
            image = cv2.imread(imagefile).astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except:
            print('filename = ' + filename)
            raise
        image, factor = ensure_max_size(image, self.opts.max_image_size)
        img_height, img_width, _ = image.shape
        labelfile = os.path.join(dirann, filename + '.txt')
        bboxes = []
        with open(labelfile, 'r') as fid:
            content = fid.read().splitlines()
            for line in content:
                line_split = line.split(' ')
                classid = int(line_split[0])
                # Get coordinates from string:
                xmin = int(line_split[1]) * factor
                ymin = int(line_split[2]) * factor
                width = int(line_split[3]) * factor
                height = int(line_split[4]) * factor
                # Ensure coordinates fit in the image size:
                xmin = max(min(xmin, img_width-2), 0)
                ymin = max(min(ymin, img_height-2), 0)
                width = max(min(width, img_width-1-xmin), 1)
                height = max(min(height, img_height-1-ymin), 1)
                # Make relative coordinates:
                xmin = xmin / img_width
                ymin = ymin / img_height
                width = width / img_width
                height = height / img_height
                bboxes.append([classid, xmin, ymin, width, height, 1.0])
        bboxes_array = np.zeros((len(bboxes), 6), dtype=np.float32)
        for i in range(len(bboxes)):
            bboxes_array[i, :] = bboxes[i]
        return image, bboxes_array


# ----------------------------------------------------------------------------------------------------------------------
# This is done to avoid memory problems.
def ensure_max_size(image, max_size):
    img_height, img_width, _ = image.shape
    factor = np.sqrt(max_size * max_size / (img_height * img_width))

    if factor < 1:
        new_width = int(img_width * factor)
        new_height = int(img_height * factor)
        image = cv2.resize(image, (new_width, new_height))
    else:
        factor = 1

    return image, factor
