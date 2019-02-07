# ======================================================================================================================
import tensorflow as tf
import tools
import logging
import os
import cv2
import numpy as np
import Preprocessor
import sys
import DataAugmentation
import Resizer
from BoundingBoxes import BoundingBox


def get_n_classes(args):
    dirdata = os.path.join(args.root_of_datasets, args.dataset_name)
    img_extension, classnames = tools.process_dataset_config(os.path.join(dirdata, 'dataset_info.xml'))
    nclasses = len(classnames)
    return nclasses


# ======================================================================================================================
class TrainDataReader:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, args, network):

        self.network = network
        self.batch_size = args.batch_size
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.num_workers = args.num_workers
        self.buffer_size = args.buffer_size
        self.preprocessor = Preprocessor.Preprocessor(self.input_width, self.input_height)

        self.resize_function = Resizer.ResizerWithLabels(self.input_width, self.input_height).get_resize_func(args.resize_method)

        self.percent_of_data = args.percent_of_data
        self.max_image_size = args.max_image_size
        self.nimages_train = None
        self.nimages_val = None
        self.train_init_op = None
        self.val_init_op = None
        self.dirdata = os.path.join(args.root_of_datasets, args.dataset_name)
        self.img_extension, self.classnames = tools.process_dataset_config(os.path.join(self.dirdata, 'dataset_info.xml'))
        self.img_extension = '.' + self.img_extension
        self.nclasses = len(self.classnames)
        self.outdir = args.outdir
        self.write_network_input = args.write_network_input

        self.shuffle_data = args.shuffle_data

        if self.img_extension == '.jpg' or self.img_extension == '.JPEG':
            self.parse_function = parse_jpg
        elif self.img_extension == '.png':
            self.parse_function = parse_png
        else:
            raise Exception('Images format not recognized.')

        self.data_aug_opts = args.data_aug_opts

        if self.data_aug_opts.apply_data_augmentation:
            data_augmenter = DataAugmentation.DetectionDataAugmentation(args, self.input_width, self.input_height, self.nclasses)
            self.data_aug_func = data_augmenter.data_augmenter
        return

    # ------------------------------------------------------------------------------------------------------------------
    def get_nbatches_per_epoch(self, split):

        if split == 'train':
            return self.nimages_train / self.batch_size
        elif split == 'val':
            return self.nimages_val / self.batch_size
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

        self.read_count = 0

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

        filenames = self.get_detection_filenames(split)
        batched_dataset = self.build_detection_dataset(filenames, split)

        return batched_dataset, len(filenames)


    # ------------------------------------------------------------------------------------------------------------------
    def get_detection_filenames(self, split):

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
        if self.percent_of_data != 100:
            # Remove data:
            indexes = np.random.choice(np.arange(len(filenamesnoext)), int(self.percent_of_data / 100.0 * len(filenamesnoext)), replace=False)
        else:
            # Shuffle data at least:
            indexes = np.arange(len(filenamesnoext))
            if self.shuffle_data:
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
    

    # ------------------------------------------------------------------------------------------------------------------
    def build_detection_dataset(self, filenames, split):
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_detection_w_all, num_parallel_calls=self.num_workers)
        if split == 'train' and self.data_aug_opts.apply_data_augmentation:
            dataset = dataset.map(self.data_aug_func, num_parallel_calls=self.num_workers)
        dataset = dataset.map(self.resize_func_extended_detection, num_parallel_calls=self.num_workers)
        dataset = dataset.map(self.preprocess_w_all, num_parallel_calls=self.num_workers)
        if self.write_network_input:
            dataset = dataset.map(self.write_network_input_func, num_parallel_calls=self.num_workers)
        dataset = dataset.map(self.encode_boxes, num_parallel_calls=self.num_workers)
        if self.shuffle_data:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        return dataset.batch(self.batch_size)


    # ------------------------------------------------------------------------------------------------------------------
    def encode_boxes(self, image, label, filename):
        (label) = tf.py_func(self.encode_boxes_np, [label, filename], (tf.float32), name='encode_boxes_np')
        label.set_shape(self.network.encoded_gt_shape)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def resize_func_extended_detection(self, image, label, filename):
        image, label = self.resize_function(image, label)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def parse_detection_w_all(self, filename):
        (image, label) = tf.py_func(self.read_detection_image_w_ann_txt, [filename], (tf.float32, tf.float32), name='read_det_im_ann')
        label.set_shape((None, 6))  # (nboxes, [class_id, x_min, y_min, width, height])
        image.set_shape((None, None, 3))  # (height, width, channels)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def preprocess_w_all(self, image, label, filename):
        image = self.preprocessor.subtract_mean(image)
        return image, label, filename


    # ------------------------------------------------------------------------------------------------------------------
    def encode_boxes_np(self, boxes_array, filename):
        nboxes = boxes_array.shape[0]
        bboxes = []
        for i in range(nboxes):
            class_id = int(np.round(boxes_array[i, 0]))
            percent_contained = boxes_array[i, 5]
            bboxes.append(BoundingBox([boxes_array[i, 1], boxes_array[i, 2], boxes_array[i, 3], boxes_array[i, 4]], class_id, percent_contained))
        encoded_label = self.network.encode_gt(bboxes, filename)
        return encoded_label


    # ------------------------------------------------------------------------------------------------------------------
    def read_detection_image_w_ann_txt(self, filename):
        # print('read_detection_image_w_ann_txt')
        self.read_count += 1

        dirimg = os.path.join(self.dirdata, "images")
        dirann = os.path.join(self.dirdata, "annotations")
        # print('antes decoding')
        filename = filename.decode(sys.getdefaultencoding())
        try:
            # print(filename)
            imagefile = os.path.join(dirimg, filename + self.img_extension)
            # print('imagefile: ' + imagefile)
            image = cv2.imread(imagefile).astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except:
            print('filename = ' + filename)
            raise
        image, factor = ensure_max_size(image, self.max_image_size)
        # print('factor = ' + str(factor))
        img_height, img_width, _ = image.shape
        # print('image.shape: ' + str([img_height, img_width]))
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
                # print('coordinates from string: ' + str([xmin, ymin, width, height]))
                # Ensure coordinates fit in the image size:
                # xmin = max(min(xmin, img_width-2), 0)
                # ymin = max(min(ymin, img_height-2), 0)
                # width = max(min(width, img_width-1-xmin), 1)
                # height = max(min(height, img_height-1-ymin), 1)
                # print('coordinates fit: ' + str([xmin, ymin, width, height]))
                # TODO: This is just to emulate the Keras SSD repository:
                # Make coordinates as integers in the 300x300 image:
                xmin_big = xmin
                ymin_big = ymin
                xmin = np.round(xmin_big * 300.0 / img_width).astype(np.int32)
                ymin = np.round(ymin_big * 300.0 / img_height).astype(np.int32)
                xmax = np.round((xmin_big + width) * 300.0 / img_width).astype(np.int32)
                ymax = np.round((ymin_big + height) * 300.0 / img_height).astype(np.int32)
                width = xmax - xmin
                height = ymax - ymin

                # Ensure the coordinates fit in the image, and that the width and height are not zero:
                width = max(width, 1)
                height = max(height, 1)
                xmin = min(xmin, 300 - width)
                ymin = min(ymin, 300 - height)

                # print('coordinates in 300x300: ' + str([xmin, ymin, width, height]))
                # Make relative coordinates:
                # xmin = xmin / img_width
                # ymin = ymin / img_height
                # width = width / img_width
                # height = height / img_height
                xmin = xmin / 300.0
                ymin = ymin / 300.0
                width = width / 300.0
                height = height / 300.0
                # print('coordinates relative: ' + str([xmin, ymin, width, height]))
                # Add as GroundTruthBox:
                # bboxes.append(BoundingBox([xmin, ymin, width, height], classid))
                # Use center coordinates:
                # xc = xmin + width / 2.0
                # yc = ymin + height / 2.0
                # xmax = xmax / 300.0
                # ymax = ymax / 300.0
                # xc = (xmin + xmax) / 2.0
                # yc = (ymin + ymax) / 2.0
                # print('coordinates center: ' + str([xc, yc, width, height]))
                # bboxes.append([classid, xc, yc, width, height])
                bboxes.append([classid, xmin, ymin, width, height, 1.0])

        #encoded_label = encode_yolo_label(bboxes, img_width, img_height)
        # encoded_label = ssd.encode_gt(bboxes)
        bboxes_array = np.zeros((len(bboxes), 6), dtype=np.float32)
        for i in range(len(bboxes)):
            bboxes_array[i, :] = bboxes[i]
        # print('read_detection_image_w_ann_txt listo')
        return image, bboxes_array


    def write_network_input_pyfunc(self, image, bboxes):
        img = image.copy()
        height = img.shape[0]
        width = img.shape[1]
        min_val = np.min(img)
        img = img - min_val
        max_val = np.max(img)
        img = img / float(max_val) * 255.0
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in bboxes:
            class_id = int(box[0])
            xmin = int(np.round(box[1] * width))
            ymin = int(np.round(box[2] * height))
            w = int(np.round(box[3] * width))
            h = int(np.round(box[4] * height))
            cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
            cv2.rectangle(img, (xmin, ymin - 20),
                          (xmin + w, ymin), (125, 125, 125), -1)
            cv2.putText(img, self.classnames[class_id], (xmin + 5, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        number = 0
        file_path_candidate = os.path.join(self.outdir, 'input' + str(number) + '.png')
        while os.path.exists(file_path_candidate):
            number += 1
            file_path_candidate = os.path.join(self.outdir, 'input' + str(number) + '.png')
        cv2.imwrite(file_path_candidate, img)
        return image


    def write_network_input_func(self, image, bboxes, filename):
        shape = image.shape
        image = tf.py_func(self.write_network_input_pyfunc, [image, bboxes], tf.float32, name='write_network_input')
        image.set_shape(shape)
        return image, bboxes, filename


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
