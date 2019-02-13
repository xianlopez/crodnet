import os
import tensorflow as tf
import cv2
import numpy as np
import Resizer
import tools
import logging


class InteractiveDataReader:

    def __init__(self, input_width, input_height, resize_method):
        self.input_width = input_width
        self.input_height = input_height
        self.resize_method = resize_method
        vgg_mean = [123.68, 116.78, 103.94]
        self.mean = np.zeros(shape=(self.input_height, self.input_width), dtype=np.float32)
        for i in range(3):
            self.mean[:, :, i] = vgg_mean[i]

    def build_inputs(self):
        return tf.placeholder(dtype=tf.float32, shape=(None, self.input_width, self.input_height, 3), name='inputs')

    def get_batch(self, image_paths):
        batch_size = len(image_paths)
        inputs_numpy = np.zeros(shape=(batch_size, self.input_width, self.input_height, 3), dtype=np.float32)
        for i in range(batch_size):
            inputs_numpy[i, :, :, :] = self.get_image(image_paths[i])
        return inputs_numpy

    def preprocess_image(self, image):
        # Resize:
        image = Resizer.ResizeNumpy(image, self.resize_method, self.input_width, self.input_height)
        # Subtract mean:
        image = image - self.mean
        return image

    def get_image(self, image_path):
        # Read image:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Preprocess it:
        image = self.preprocess_image(image)
        return image


###----------------------------------------------------------------------------###
class InteractiveDataReaderFromDataset(InteractiveDataReader):

    def __init__(self, input_width, input_height, opts, multi_cell_arch, batch_size, split):
        super(InteractiveDataReaderFromDataset, self).__init__(input_width, input_height, opts.resize_method)
        self.multi_cell_arch = multi_cell_arch
        self.opts = opts
        self.batch_size = batch_size
        self.dirdata = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name)
        dataset_info_path = os.path.join(self.dirdata, 'dataset_info.xml')
        self.img_extension, self.classnames = tools.process_dataset_config(dataset_info_path)
        self.filenames = self.get_filenames(split)

    def get_batch(self, image_paths):
        pass

    def image_path_2_label_path(self):
        pass

    def get_label(self, image_path):
        pass

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
        if len(filenamesnoext) % self.batch_size != 0:
            aux = filenamesnoext
            filenamesnoext = []
            for i in range(len(aux) - (len(aux) % self.batch_size)):
                filenamesnoext.append(aux[i])
        assert len(filenamesnoext) % self.batch_size == 0, 'Number of images is not a multiple of batch size'
        return filenamesnoext
