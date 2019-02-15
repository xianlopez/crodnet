import os
import numpy as np
import sys
import cv2
import tools
import logging


class CommonDataReader:
    def __init__(self, opts):
        self.dirdata = os.path.join(opts.root_of_datasets, opts.dataset_name)
        self.img_extension, self.classnames = tools.process_dataset_config(os.path.join(self.dirdata, 'dataset_info.xml'))
        self.img_extension = '.' + self.img_extension
        self.max_image_size = opts.max_image_size
        self.percent_of_data = opts.percent_of_data
        self.shuffle_data = opts.shuffle_data
        self.n_images_per_batch = opts.single_cell_opts.n_images_per_batch

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
        image, factor = ensure_max_size(image, self.max_image_size)
        img_height, img_width, _ = image.shape
        labelfile = os.path.join(dirann, filename + '.txt')
        bboxes = []
        gt_idx = -1
        with open(labelfile, 'r') as fid:
            content = fid.read().splitlines()
            for line in content:
                gt_idx += 1
                line_split = line.split(' ')
                classid = int(line_split[0])
                # Get coordinates from string:
                xmin = int(line_split[1]) * factor
                ymin = int(line_split[2]) * factor
                width = int(line_split[3]) * factor
                height = int(line_split[4]) * factor
                # Ensure coordinates fit in the image size:
                xmin = max(min(xmin, img_width - 2), 0)
                ymin = max(min(ymin, img_height - 2), 0)
                width = max(min(width, img_width - 1 - xmin), 1)
                height = max(min(height, img_height - 1 - ymin), 1)
                # Make relative coordinates:
                xmin = xmin / img_width
                ymin = ymin / img_height
                width = width / img_width
                height = height / img_height
                bboxes.append([classid, xmin, ymin, width, height, 1.0, gt_idx])
        bboxes_array = np.zeros((len(bboxes), 7), dtype=np.float32)
        for i in range(len(bboxes)):
            bboxes_array[i, :] = bboxes[i]
        return image, bboxes_array

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
        if self.percent_of_data != 100:
            # Remove data:
            indexes = np.random.choice(np.arange(len(filenamesnoext)),
                                       int(self.percent_of_data / 100.0 * len(filenamesnoext)), replace=False)
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
        if len(filenamesnoext) % self.n_images_per_batch != 0:
            aux = filenamesnoext
            filenamesnoext = []
            for i in range(len(aux) - (len(aux) % self.n_images_per_batch)):
                filenamesnoext.append(aux[i])
        assert len(filenamesnoext) % self.n_images_per_batch == 0, 'Number of images is not a multiple of n_images_per_batch'
        return filenamesnoext


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