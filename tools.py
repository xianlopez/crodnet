# ======================================================================================================================
import time
import os
import logging
import sys
from shutil import copyfile
import numpy as np
import cv2
from lxml import etree
import tensorflow as tf
import scipy.misc
import matplotlib
matplotlib.use('Agg') # To avoid exception 'async handler deleted by the wrong thread'
from matplotlib import pyplot as plt
import operator
import BoundingBoxes
import Resizer


# ----------------------------------------------------------------------------------------------------------------------
batch_size = None
def get_batch_size():
    if batch_size is None:
        raise Exception('batch size used before set.')
    else:
        return batch_size


# ----------------------------------------------------------------------------------------------------------------------
repository_dir = None
def get_repository_dir():
    if repository_dir is None:
        global repository_dir
        repository_dir = os.path.dirname(os.path.abspath(__file__))
        return repository_dir
    else:
        return repository_dir


# ----------------------------------------------------------------------------------------------------------------------
def adapt_path_to_current_os(path):
    if os.sep == '\\': # Windows
        path = path.replace('/', os.sep)
    else: # Linux
        path = path.replace('\\', os.sep)
    return path


# ----------------------------------------------------------------------------------------------------------------------
def join_paths(*args):

    if len(args) <= 1:
        raise Exception('Not enough arguments')
    fullpath = args[0]
    for i in range(1, len(args)):
        # fullpath = fullpath + '\\' + args[i]
        fullpath = os.path.join(fullpath, args[i])

    return fullpath


# ----------------------------------------------------------------------------------------------------------------------
def get_checkpoint(args):

    if args.checkpoint_step == 'last':
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint = os.path.join(args.checkpoint_dir, 'model-' + str(args.checkpoint_step))

    return checkpoint


# ----------------------------------------------------------------------------------------------------------------------
def process_dataset_config(dataset_info_path):

    dataset_config_file = os.path.join(dataset_info_path)
    tree = etree.parse(dataset_config_file)
    root = tree.getroot()
    images_format = root.find('format').text
    classes = root.find('classes')
    classnodes = classes.findall('class')
    classnames = [''] * len(classnodes)

    for cn in classnodes:
        classid = cn.find('id').text
        name = cn.find('name').text
        assert classid.isdigit(), 'Class id must be a non-negative integer.'
        assert int(classid) < len(classnodes), 'Class id greater or equal than classes number.'
        classnames[int(classid)] = name

    for i in range(len(classnames)):
        assert classnames[i] != '', 'Name not found for id ' + str(i)

    return images_format, classnames

# ----------------------------------------------------------------------------------------------------------------------
def plot_training_history(train_metrics, train_loss, val_metrics, val_loss, metric_names, args, epoch_num):
    metrics_colors = ['r', 'g', 'y', 'm', 'c']
    if len(train_loss) >= 2 or len(val_loss) >= 2:

        # Epochs on which we computed train and validation measures:
        x_train = np.arange(args.nepochs_checktrain, epoch_num + 1, args.nepochs_checktrain)
        x_val = np.arange(args.nepochs_checkval, epoch_num + 1, args.nepochs_checkval)
        # Initialize figure:
        # Axis 1 will be for metrics, and axis 2 for losses.
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        if len(train_loss) >= 2:
            # Train loss:
            ax2.plot(x_train, train_loss, 'b-', label='train loss')
            # Train metrics:
            for m_idx in range(len(metric_names)):
                this_metric = []
                for i in range(len(train_metrics)):
                    this_metric.append(train_metrics[i][m_idx])
                ax1.plot(x_train, this_metric, metrics_colors[m_idx] + '-', label='train ' + metric_names[m_idx])
        if len(val_loss) >= 2:
            # Val loss:
            ax2.plot(x_val, val_loss, 'b--', label='val loss')
            # Val metric:
            for m_idx in range(len(metric_names)):
                this_metric = []
                for i in range(len(val_metrics)):
                    this_metric.append(val_metrics[i][m_idx])
                ax1.plot(x_val, this_metric, metrics_colors[m_idx] + '--', label='val ' + metric_names[m_idx])

        # Axis limits for metrics:
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, np.max(np.concatenate((train_loss, val_loss))))

        # Add title
        plt.title('Train history')

        # Add axis labels
        ax1.set_ylabel('Metrics')
        ax2.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')

        # To adjust correctly everything:
        fig.tight_layout()

        # Add legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')

        # Delete previous figure to save the new one
        fig_path = os.path.join(args.outdir, 'train_history.png')
        if os.path.exists(fig_path):
            try:
                os.remove(fig_path)
            except:
                logging.warning('Error removing ' + fig_path + '. Using alternative name.')
                fig_path = os.path.join(args.outdir, 'train_history_' + str(epoch_num) + '.png')

        # Save fig
        plt.savefig(fig_path)

        # Close plot
        plt.close()

    return


# ----------------------------------------------------------------------------------------------------------------------
def write_results(all_predictions, all_labels, all_names, classnames, args, input_width, input_height, action, img_extension):

    if args.write_images or args.write_results:
        logging.info('Writing results...')
        write_detection_results(all_predictions, all_labels, all_names, classnames, args, input_width, input_height, action, img_extension)
        logging.info('Results written.')

    return


# ----------------------------------------------------------------------------------------------------------------------
def show_hard_negative_mask(predictions, labels, hnm, filenames, classnames, args, img_extension):
    nimages = len(predictions)

    dirimg = os.path.join(args.root_of_datasets, args.dataset_name, "images")
    filepaths = []
    for i in range(nimages):
        name = filenames[i].decode(sys.getdefaultencoding())
        filepaths.append(os.path.join(dirimg, name + img_extension))

    # Loop on images:
    for i in range(nimages):
        _, filename = os.path.split(filepaths[i])
        rawname, _ = os.path.splitext(filename)
        img = cv2.imread(filepaths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        orig_height, orig_width, _ = img.shape
        draw_labels_and_defaults(img, predictions[i], classnames, labels[i], hnm[i])

    return


# ----------------------------------------------------------------------------------------------------------------------
def draw_labels_and_defaults(img, boxes, class_name, labels, hnm):
    # boxes: List with so many elements as bounding boxes.
    # Each element has the following content: [class_index, x0, y0, w, h, confidence]
    # Draw ground truth:
    for box in hnm:
        [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
        cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (127, 0, 127), 2)
    for box in labels:
        [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
        cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    cv2.imshow('image', img2)
    cv2.waitKey(0)
    # cv2.imwrite(r'D:\image.png', img)
    return


# ----------------------------------------------------------------------------------------------------------------------
def write_detection_results(predictions, labels, filenames, classnames, args, input_width, input_height, action, img_extension):
    nimages = len(predictions)

    # If we are in prediction mode, the filepaths have the whole paths of the images. Otherwise, they are relative
    # paths, so we have to add the dataset directory:
    # In any case, we have to decode the strings.
    if action == 'predict':
        filepaths = filenames
        for i in range(nimages):
            filepaths[i] = filepaths[i].decode(sys.getdefaultencoding())
    else:
        dirimg = os.path.join(args.root_of_datasets, args.dataset_name, "images")
        filepaths = []
        for i in range(nimages):
            name = filenames[i].decode(sys.getdefaultencoding())
            filepaths.append(os.path.join(dirimg, name + img_extension))

    # Create folders, if we are going to write results of images:
    if args.write_images:
        os.makedirs(os.path.join(args.outdir, 'images'))
    if args.write_results:
        os.makedirs(os.path.join(args.outdir, 'results'))

    # Loop on images:
    for i in range(nimages):

        _, filename = os.path.split(filepaths[i])
        rawname, _ = os.path.splitext(filename)
        results = predictions[i]
        img = cv2.imread(filepaths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = Resizer.ResizeNumpy(img, args.resize_method, input_width, input_height)
        orig_height, orig_width, _ = img.shape
        if args.write_images:
            if labels is None:
                draw_result(img, results, classnames, None)
            else:
                draw_result(img, results, classnames, labels[i])
            newpath = os.path.join(args.outdir, 'images', filename + img_extension)
            cv2.imwrite(newpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Write the coordinates of the bounding boxes to txt files:
        if args.write_results:
            write_boxes_to_txt(results, filename, args, img)

    return


# ----------------------------------------------------------------------------------------------------------------------
def write_boxes_to_txt(boxes, imagename, args, img):
    rawname, _ = os.path.splitext(imagename)
    txt_file = os.path.join(args.outdir, 'results', rawname) + '.txt'
    with open(txt_file, 'w') as fid:
        for box in boxes:
            coords = box.get_abs_coords_cv(img)
            fid.write(str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + ' ' + str(coords[3]) + ' ' + str(box.classid) + '\n')
    return


# ----------------------------------------------------------------------------------------------------------------------
def compute_iou(box1, box2):
    # box coordinates: [xmin, ymin, w, h]
    if np.min(np.array(box1[2:])) < 0 or np.min(np.array(box2[2:])) < 0:
        # We make sure width and height are non-negative. If that happens, just assign 0 iou.
        iou = 0
    else:
        lu = np.max(np.array([box1[:2], box2[:2]]), axis=0)
        rd = np.min(np.array([[box1[0] + box1[2], box1[1] + box1[3]], [box2[0] + box2[2], box2[1] + box2[3]]]), axis=0)
        intersection = np.maximum(0.0, rd-lu)
        intersec_area = intersection[0] * intersection[1]
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        if area1 < 1e-6 or area2 < 1e-6:
            iou = 0
        else:
            union_area = area1 + area2 - intersec_area
            iou = intersec_area / np.float(union_area)
    return iou


# ----------------------------------------------------------------------------------------------------------------------
def compute_fraction_contained(box_small, box_big):
    # box coordinates: [xmin, ymin, w, h]
    if np.min(np.array(box_small[2:])) < 0 or np.min(np.array(box_big[2:])) < 0:
        # We make sure width and height are non-negative. If that happens, just assign 0 fraction_contained.
        fraction_contained = 0
    else:
        lu = np.max(np.array([box_small[:2], box_big[:2]]), axis=0)
        rd = np.min(np.array([[box_small[0] + box_small[2], box_small[1] + box_small[3]], [box_big[0] + box_big[2], box_big[1] + box_big[3]]]), axis=0)
        intersection = np.maximum(0.0, rd-lu)
        intersec_area = intersection[0] * intersection[1]
        area_box_small = box_small[2] * box_small[3]
        fraction_contained = intersec_area / np.float(area_box_small)
    return fraction_contained


# ----------------------------------------------------------------------------------------------------------------------
def add_bounding_boxes_to_image(image, bboxes, color=(0,0,255), line_width=2):
    # bboxes: (nboxes, 5) [class_id, xmin, ymin, width, height] in relative coordinates
    height = image.shape[0]
    width = image.shape[1]
    for i in range(bboxes.shape[0]):
        xmin = int(np.round(bboxes[i, 1] * width))
        ymin = int(np.round(bboxes[i, 2] * height))
        w = int(np.round(bboxes[i, 3] * width))
        h = int(np.round(bboxes[i, 4] * height))
        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, line_width)
    return image


# ----------------------------------------------------------------------------------------------------------------------
def add_bounding_boxes_to_image2(image, bboxes, color=(0,0,255), line_width=2):
    # bboxes: List of BoundingBox objects
    height = image.shape[0]
    width = image.shape[1]
    for box in bboxes:
        abs_coords = box.get_abs_coords(width, height)
        xmin = abs_coords[0]
        ymin = abs_coords[1]
        w = abs_coords[2]
        h = abs_coords[3]
        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, line_width)
    return image


# ----------------------------------------------------------------------------------------------------------------------
def add_bounding_boxes_to_image_with_class_names(image, bboxes, classnames, color=(0,0,255), line_width=2):
    # bboxes: List of BoundingBox objects
    height = image.shape[0]
    width = image.shape[1]
    for box in bboxes:
        abs_coords = box.get_abs_coords(width, height)
        xmin = abs_coords[0]
        ymin = abs_coords[1]
        w = abs_coords[2]
        h = abs_coords[3]
        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, line_width)
        cv2.rectangle(image, (xmin, ymin - 20), (xmin + w, ymin), (125, 125, 125), -1)
        cv2.putText(image, classnames[box.classid], (xmin + 5, ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


# ----------------------------------------------------------------------------------------------------------------------
def non_maximum_suppression(boxes, threshold_nms):
    # boxes: List with all the predicted bounding boxes in the image.
    nboxes = len(boxes)
    boxes.sort(key=operator.attrgetter('confidence'), reverse=True)
    for i in range(nboxes):
        if boxes[i].confidence != -np.inf:
            for j in range(i + 1, nboxes):
                if boxes[j].confidence != -np.inf:
                    if compute_iou(boxes[i].get_coords(), boxes[j].get_coords()) > threshold_nms:
                        boxes[j].confidence = -np.inf
    remaining_boxes = [x for x in boxes if x.confidence != -np.inf]
    return remaining_boxes


# ----------------------------------------------------------------------------------------------------------------------
def convert_boxes_to_original_size(boxes, orig_width, orig_height, input_width, input_height):
    newboxes = boxes[:]
    for i in range(len(boxes)):
        newboxes[i].xmin *= (1.0 * orig_width / input_width)
        newboxes[i].ymin *= (1.0 * orig_height / input_height)
        newboxes[i].width *= (1.0 * orig_width / input_width)
        newboxes[i].height *= (1.0 * orig_height / input_height)
    return newboxes


# ----------------------------------------------------------------------------------------------------------------------
def draw_result(img, boxes, class_name, labels=None):
    # boxes: List with so many elements as bounding boxes.
    # Each element has the following content: [class_index, x0, y0, w, h, confidence]
    # Draw ground truth:
    if labels is not None:
        for box in labels:
            [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
            cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
    # Draw predictions:
    for box in boxes:
        conf = box.confidence
        [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
        classid = int(box.classid)
        # Select color depending if the prediction is a true positive or not:
        if box.tp == 'yes':
            color = (0, 255, 0)
        elif box.tp == 'no':
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        # Draw box:
        cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), color, 2)
        cv2.rectangle(img, (xmin, ymin - 20),
                      (xmin + w, ymin), (125, 125, 125), -1)
        cv2.putText(img, class_name[classid] + ' : %.2f' % conf, (xmin + 5, ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return


# ----------------------------------------------------------------------------------------------------------------------
def create_experiment_folder(args):
    year = time.strftime('%Y')
    month = time.strftime('%m')
    day = time.strftime('%d')
    if not os.path.exists(args.experiments_folder):
        os.mkdir(args.experiments_folder)
    year_folder = join_paths(args.experiments_folder, year)
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)
    base_name = join_paths(year_folder, year + '_' + month + '_' + day)
    experiment_folder = base_name
    count = 0
    while os.path.exists(experiment_folder):
        count += 1
        experiment_folder = base_name + '_' + str(count)
    os.mkdir(experiment_folder)
    print('Experiment folder: ' + experiment_folder)
    return experiment_folder


# ----------------------------------------------------------------------------------------------------------------------
def copy_config(args, inline_args):
    if inline_args.run == 'train':
        configModuleName = 'train_config'
    elif inline_args.run == 'evaluate':
        configModuleName = 'eval_config'
    elif inline_args.run == 'predict':
        configModuleName = 'predict_config'
    else:
        print('Please, select specify a valid execution mode: train / evaluate / predict')
        raise Exception()

    if inline_args.conf is not None:
        configModuleName = configModuleName + '_' + inline_args.conf
        configModuleNameAndPath = os.path.join('config', configModuleName)
    else:
        configModuleNameAndPath = configModuleName

    configModuleNameAndPath = os.path.join(get_repository_dir(), configModuleNameAndPath + '.py')

    copyfile(configModuleNameAndPath, os.path.join(args.outdir, configModuleName + '.py'))
    return


# ----------------------------------------------------------------------------------------------------------------------
def configure_logging(args):
    if len(logging.getLogger('').handlers) == 0:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=join_paths(args.outdir, 'out.log'),
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler(sys.stdout)
        # console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        file_handler = None
        for handler in logging.getLogger('').handlers:
            if type(handler) == logging.FileHandler:
                file_handler = handler
        if file_handler is None:
            raise Exception('File handler not found.')
        logging.getLogger('').removeHandler(file_handler)
        fileh = logging.FileHandler(filename=join_paths(args.outdir, 'out.log'), mode='w')
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
        fileh.setFormatter(formatter)
        fileh.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(fileh)
    logging.info('Logging configured.')


# ----------------------------------------------------------------------------------------------------------------------
def get_config_proto(gpu_memory_fraction):
    if gpu_memory_fraction > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    else:
        gpu_options = tf.GPUOptions()
    return tf.ConfigProto(gpu_options=gpu_options)


# ----------------------------------------------------------------------------------------------------------------------
def process_grid_detection(boxes_grid, conf, threshold):
    # boxes_grid: (grid_size, grid_size, 4)
    # conf: (grid_size, grid_size, nclasses)
    grid_size, _, _ = boxes_grid.shape
    assert boxes_grid.shape[1] == grid_size
    assert conf.shape[0] == grid_size, "Wrong dimensionality"
    assert conf.shape[1] == grid_size, "Wrong dimensionality"
    results = []
    for i in range(grid_size):
        for j in range(grid_size):
            max_conf = np.max(conf[i, j, :])
            if max_conf > threshold:
                results.append([np.argmax(conf[i, j, :]), boxes_grid[i, j, 0], boxes_grid[i, j, 1],
                                boxes_grid[i, j, 2], boxes_grid[i, j, 3], max_conf])
    return results


# ----------------------------------------------------------------------------------------------------------------------
def save_input_images(names, images, args, epoch_num, batch_num, img_extension, labels=None):
    input_imgs_dir = os.path.join(args.outdir, 'input_images')
    if not os.path.exists(input_imgs_dir):
        os.makedirs(input_imgs_dir)
    epoch_dir = os.path.join(input_imgs_dir, 'epoch_'+str(epoch_num))
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    batch_dir = os.path.join(epoch_dir, 'batch_'+str(batch_num))
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    for i in range(len(names)):
        filename = names[i].decode(sys.getdefaultencoding())
        # Remove drive:
        if ':' in filename:
            _, filename = os.path.split(filename)
        # Split subfolders:
        subfolders = filename.split(os.sep)
        img_folder = batch_dir
        for j in range(len(subfolders) - 1):
            img_folder = os.path.join(img_folder, subfolders[j])
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        # Final name and path:
        filename = subfolders[len(subfolders) - 1]
        img_path = os.path.join(img_folder, filename)
        # print(img_path)
        if img_path[len(img_path)-4:] != img_extension:
            img_path = img_path + img_extension
        image_to_write = images[i]
        max_value = np.max(image_to_write)
        if labels is not None:
            for box in labels[i]:
                x0 = int(box.x0)
                y0 = int(box.y0)
                x1 = min(x0 + int(box.width), image_to_write.shape[1]-1)
                y1 = min(y0 + int(box.height), image_to_write.shape[0]-1)
                image_to_write[y0, x0:x1, 0] = max_value
                image_to_write[y1, x0:x1, 0] = max_value
                image_to_write[y0:y1, x0, 0] = max_value
                image_to_write[y0:y1, x1, 0] = max_value
        scipy.misc.imsave(img_path, image_to_write)
    return


# ----------------------------------------------------------------------------------------------------------------------
def keep_last_part_of_path(full_path, first_part):
    subfolders_full = full_path.split(os.sep)
    subfolders_first = first_part.split(os.sep)
    if len(subfolders_full) <= len(subfolders_first):
        raise Exception('Length of full_path is lower or equal to length of first_part')
    for i in range(len(subfolders_first)):
        if subfolders_full[i] != subfolders_first[i]:
            print('full_path: ' + full_path)
            print('first_part: ' + first_part)
            raise Exception('first_part not contained in full_path')
    last_part = subfolders_full[len(subfolders_first)]
    if len(subfolders_full) > len(subfolders_first) + 1:
        for i in range(len(subfolders_first) + 1, len(subfolders_full)):
            last_part = os.path.join(last_part, subfolders_full[i])
    return last_part


# ----------------------------------------------------------------------------------------------------------------------
def ensure_new_path(path_in):
    dot_pos = path_in.rfind('.')
    rawname = path_in[:dot_pos]
    extension = path_in[dot_pos:]
    new_path = path_in
    count = 0
    while os.path.exists(new_path):
        count += 1
        new_path = rawname + '_' + str(count) + extension
    return new_path


# ----------------------------------------------------------------------------------------------------------------------
def get_trainable_variables(args):
    # Choose the variables to train:
    if args.layers_list is None or args.layers_list == []:
        # Train all variables
        vars_to_train = tf.trainable_variables()

    else:
        # Train the variables included in layers_list
        if args.train_selected_layers:

            vars_to_train = []
            for v in tf.trainable_variables():
                selected = False
                for layer in args.layers_list:
                    if layer in v.name:
                        selected = True
                        break
                if selected:
                    vars_to_train.append(v)

        # Train the variables NOT included in layers_list
        else:

            vars_to_train = []
            for v in tf.trainable_variables():
                selected = True
                for layer in args.layers_list:
                    if layer in v.name:
                        selected = False
                        break
                if selected:
                    vars_to_train.append(v)
    return vars_to_train


def get_kernel_sides_and_pads(kernel, pad_mode):
    kernel_minus = int(np.floor((kernel - 1) / 2.0))
    kernel_plus = int(np.ceil((kernel - 1) / 2.0))
    if pad_mode.lower() == 'valid':
        pad_minus = 0
        pad_plus = 0
    elif pad_mode.lower() == 'same':
        pad_minus = kernel_minus
        pad_plus = kernel_plus
    else:
        raise Exception('Pad-mode not recognized.')
    return kernel_minus, kernel_plus, pad_minus, pad_plus


def receptive_field_one_layer_1d_0based(position, stride, kernel, pad_mode):
    # position is 0-based (starts in 0, not in 1).
    kernel_minus, kernel_plus, pad_minus, pad_plus = get_kernel_sides_and_pads(kernel, pad_mode)
    lower_limit = int(stride * position - pad_minus)
    upper_limit = int(stride * position - pad_minus + kernel - 1)
    return [lower_limit, upper_limit]


def receptive_field_center_one_layer_1d_0based(position, stride, kernel, pad_mode):
    # position is 0-based (starts in 0, not in 1).
    kernel_minus, kernel_plus, pad_minus, pad_plus = get_kernel_sides_and_pads(kernel, pad_mode)
    receptive_field_center = stride * position - pad_minus + float(kernel - 1) / 2.0
    return receptive_field_center


def compute_receptive_field_range(positions, layers):
    # positions is a tuple of two elements, 0-based (they start in 0, not in 1).
    # layers is a list of elements [strides, kernels, pad_mode]
    n_layers = len(layers)
    this_layer = layers[n_layers - 1]
    strides = this_layer[0]
    kernels = this_layer[1]
    pad_mode = this_layer[2]
    range_row = receptive_field_one_layer_1d_0based(positions[0], strides[0], kernels[0], pad_mode)
    range_col = receptive_field_one_layer_1d_0based(positions[1], strides[1], kernels[1], pad_mode)
    if n_layers > 1:
        for i in range(n_layers - 2, -1, -1):
            this_layer = layers[i]
            strides = this_layer[0]
            kernels = this_layer[1]
            pad_mode = this_layer[2]
            range_row_lower = receptive_field_one_layer_1d_0based(range_row[0], strides[0], kernels[0], pad_mode)
            range_row_upper = receptive_field_one_layer_1d_0based(range_row[1], strides[0], kernels[0], pad_mode)
            range_col_lower = receptive_field_one_layer_1d_0based(range_col[0], strides[1], kernels[1], pad_mode)
            range_col_upper = receptive_field_one_layer_1d_0based(range_col[1], strides[1], kernels[1], pad_mode)
            range_row = [range_row_lower[0], range_row_upper[1]]
            range_col = [range_col_lower[0], range_col_upper[1]]
    return range_row, range_col


def compute_receptive_field_center(positions, layers):
    # positions is a tuple of two elements, 0-based (they start in 0, not in 1).
    # layers is a list of elements [strides, kernels, pad_mode]
    n_layers = len(layers)
    position_row = positions[0]
    position_col = positions[1]
    for i in range(n_layers - 1, -1, -1):
        this_layer = layers[i]
        strides = this_layer[0]
        kernels = this_layer[1]
        pad_mode = this_layer[2]
        position_row = receptive_field_center_one_layer_1d_0based(position_row, strides[0], kernels[0], pad_mode)
        position_col = receptive_field_center_one_layer_1d_0based(position_col, strides[1], kernels[1], pad_mode)
    return [position_row, position_col]









