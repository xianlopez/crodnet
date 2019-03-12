# ======================================================================================================================
import tensorflow as tf
import Preprocessor
import DataAugmentation
import ImageCropper
import network
from CommonDataReader import CommonDataReader
import os
import numpy as np
import sys


# ======================================================================================================================
class TrainDataReader(CommonDataReader):
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, opts, single_cell_arch):

        super(TrainDataReader, self).__init__(opts, opts.single_cell_opts.n_images_per_batch)

        self.single_cell_arch = single_cell_arch
        self.n_crops_per_image = opts.single_cell_opts.n_crops_per_image
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.num_workers = opts.num_workers
        self.buffer_size = opts.buffer_size
        self.preprocessor = Preprocessor.Preprocessor(self.input_width, self.input_height)

        self.nimages_train = None
        self.nimages_val = None
        self.train_init_op = None
        self.val_init_op = None

        self.outdir = opts.outdir
        self.hard_negatives_dir = os.path.join(self.outdir, 'hard_negatives')
        # self.hard_negatives_dir = '/home/xian/crodnet/experiments/2019/2019_03_12_13/hard_negatives'

        self.image_cropper = ImageCropper.ImageCropper(opts.image_cropper_opts, self.single_cell_arch, opts.single_cell_opts.n_crops_per_image)

        if self.img_extension == '.jpg' or self.img_extension == '.JPEG':
            self.parse_function = parse_jpg
        elif self.img_extension == '.png':
            self.parse_function = parse_png
        else:
            raise Exception('Images format not recognized.')

        self.data_aug_opts = opts.data_aug_opts

        if self.data_aug_opts.apply_data_augmentation:
            data_augmenter = DataAugmentation.DataAugmentation(opts, self.input_width, self.input_height)
            self.data_aug_func = data_augmenter.data_augmenter

        return

    # ------------------------------------------------------------------------------------------------------------------
    def get_nbatches_per_epoch(self, split):
        if split == 'train':
            return self.nimages_train / self.n_images_per_batch
        elif split == 'val':
            return self.nimages_val / self.n_images_per_batch
        else:
            raise Exception('Split not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def get_init_op(self, split):
        if split == 'train':
            return self.train_init_op
        elif split == 'val':
            return self.val_init_op
        else:
            raise Exception('Split not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def build_iterator(self):
        batched_dataset_train, self.nimages_train = self.build_batched_dataset('train')
        print('Number of training examples: ' + str(self.nimages_train))
        batched_dataset_val, self.nimages_val = self.build_batched_dataset('val')
        print('Number of validation examples: ' + str(self.nimages_val))
        iterator = tf.data.Iterator.from_structure(batched_dataset_train.output_types,
                                                   batched_dataset_train.output_shapes)
        inputs, labels, filenames = iterator.get_next(name='iterator-output')
        self.train_init_op = iterator.make_initializer(batched_dataset_train, name='train_init_op')
        self.val_init_op = iterator.make_initializer(batched_dataset_val, name='val_init_op')
        return inputs, labels, filenames

    # ------------------------------------------------------------------------------------------------------------------
    def build_batched_dataset(self, split):
        filenames = self.get_filenames(split)
        batched_dataset = self.build_dataset(filenames, split)
        return batched_dataset, len(filenames)

    # ------------------------------------------------------------------------------------------------------------------
    def build_dataset(self, filenames, split):
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_image, num_parallel_calls=self.num_workers)  # image, bboxes, hns, filename
        if split == 'train' and self.data_aug_opts.apply_data_augmentation:
            dataset = dataset.map(self.data_aug_func, num_parallel_calls=self.num_workers)  # image, bboxes, hns, filename
        dataset = dataset.map(self.preprocess_extended, num_parallel_calls=self.num_workers)  # image, bboxes, hns, filename
        dataset = dataset.map(self.take_crops_from_image, num_parallel_calls=self.num_workers)  # crops, label_enc, filename
        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        return dataset.batch(self.n_images_per_batch)

    # ------------------------------------------------------------------------------------------------------------------
    def parse_image(self, filename):
        (image, bboxes) = tf.py_func(self.read_image_with_bboxes, [filename], (tf.float32, tf.float32), name='read_det_im_ann')
        bboxes.set_shape((None, 7))  # (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        image.set_shape((None, None, 3))  # (height, width, 3)
        (hns) = tf.py_func(self.read_hard_negatives, [filename], (tf.float32), name='read_hns')
        hns.set_shape((None, 7))  # (n_hns, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        return image, bboxes, hns, filename

    def read_hard_negatives(self, filename):
        if type(filename) == bytes:
            filename = filename.decode(sys.getdefaultencoding())
        if os.path.exists(self.hard_negatives_dir):
            hns_file = os.path.join(self.hard_negatives_dir, filename + '.txt')
            if os.path.exists(hns_file):
                with open(hns_file, 'r') as fid:
                    lines = fid.read().split('\n')
                    lines = [line for line in lines if line != '']
                    hns_list = []
                    for line in lines:
                        line_split = line.split(' ')
                        classid = int(line_split[0])
                        xmin = float(line_split[1])
                        ymin = float(line_split[2])
                        width = float(line_split[3])
                        height = float(line_split[4])
                        hns_list.append([classid, xmin, ymin, width, height, 1.0, -1.0])
                    n_hns = len(hns_list)
                    hns = np.zeros((n_hns, 7), dtype=np.float32)
                    for i in range(n_hns):
                        hns[i, :] = hns_list[i]
            else:
                hns = np.zeros(shape=(0, 7), dtype=np.float32)
        else:
            hns = np.zeros(shape=(0, 7), dtype=np.float32)
        return hns

    # ------------------------------------------------------------------------------------------------------------------
    def preprocess_extended(self, crops, bboxes, hns, filename):
        # crops: (n_crops_per_image, height, width, 3)
        # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
        # filename: ()
        crops = self.preprocessor.subtract_mean_tf(crops)
        return crops, bboxes, hns, filename

    def take_crops_from_image(self, image, bboxes, hns, filename):
        # image: (height, width, 3)
        # bboxes: (n_gt, 7) [class_id, xmin, ymin, width, height, percent_contained, gt_idx]
        # filename: ()
        crops, labels_enc = tf.py_func(self.image_cropper.take_crops_on_image, [image, bboxes, hns], (tf.float32, tf.float32))
        crops.set_shape((self.n_crops_per_image, network.receptive_field_size, network.receptive_field_size, 3))
        labels_enc.set_shape((self.n_crops_per_image, self.single_cell_arch.n_labels))
        # crops, labels_enc = self.image_cropper.take_crops_on_image(image, bboxes)
        # crops: (n_crops_per_image, receptive_field_size, receptive_field_size, 3)
        # labels_enc: (n_crops_per_image, n_labels)
        return crops, labels_enc, filename


# ----------------------------------------------------------------------------------------------------------------------
def parse_jpg(filepath):
    img = tf.read_file(filepath, name='read_jpg')
    img = tf.image.decode_jpeg(img, channels=3, name='decode_jpg')
    img = tf.cast(img, tf.float32, name='cast_jpg2float32')

    return img


# ----------------------------------------------------------------------------------------------------------------------
def parse_png(filepath):
    img = tf.read_file(filepath, name='read_png')
    img = tf.image.decode_png(img, channels=3, name='decode_png')
    img = tf.cast(img, tf.float32, name='cast_png2float32')

    return img
