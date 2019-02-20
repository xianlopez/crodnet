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
import importlib
import matplotlib
matplotlib.use('Agg') # To avoid exception 'async handler deleted by the wrong thread'
from matplotlib import pyplot as plt


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
def import_config_files(inline_args, configModuleName, class2load):

    if inline_args.conf is not None:
        configModuleName = configModuleName + '_' + inline_args.conf
        configModuleNameAndPath = "config." + configModuleName
    else:
        configModuleNameAndPath = configModuleName

    try:
        currentConfiguration = getattr(importlib.import_module(configModuleNameAndPath), class2load)
    except:
        if inline_args.conf is not None:
            print('.' + os.sep + 'config' + os.sep + configModuleName + ' configuration file NOT found, or ' + class2load +
                  ' class not defined.')
        else:
            print(configModuleName + ' configuration file NOT found, or ' + class2load + ' class not defined.')
        raise

    return currentConfiguration


# ----------------------------------------------------------------------------------------------------------------------
def common_stuff(inline_args, configModuleName, class2load):

    # Set visible GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(inline_args.gpu)

    # Import appropriate config file as interactive module loading:
    currentConfiguration = import_config_files(inline_args, configModuleName, class2load)

    # Get arguments from current configuration:
    opts = currentConfiguration()

    # Set level of TensorFlow logger:
    if opts.tf_log_level == 'SILENT':
        level = 3
    elif opts.tf_log_level == 'ERROR':
        level = 2
    elif opts.tf_log_level == 'WARNING':
        level = 1
    elif opts.tf_log_level == 'INFO':
        level = 0
    else:
        err_msg = 'TensorFlow log level not understood.'
        logging.error(err_msg)
        raise Exception(err_msg)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)

    # Create experiment folder:
    if opts.experiments_folder[0] == '.':  # Relative path
        opts.experiments_folder = os.path.join(os.getcwd(), opts.experiments_folder[2:])
        opts.outdir = create_experiment_folder(opts)
    # Configure logger:
    configure_logging(opts)

    # Copy configuration file to the exeperiment folder:
    try:
        copy_config(opts, inline_args, configModuleName)
    except Exception as ex:
        err_msg = 'Error copying config file.'
        logging.error(err_msg)
        logging.error(ex)
        raise Exception(err_msg)

    # Set random seed:
    if opts.random_seed is not None:
        tf.set_random_seed(opts.random_seed)
        np.random.seed(opts.random_seed)

    return opts


# ----------------------------------------------------------------------------------------------------------------------
def add_mean_again(image):
    mean = [123.0, 117.0, 104.0]
    mean = np.reshape(mean, [1, 1, 3])
    image = image + mean
    return image


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
def add_bounding_boxes_to_image2(image, bboxes, classnames, color=(0,0,255), line_width=2):
    # bboxes: (nboxes, 6) [class_id, xmin, ymin, width, height, conf] in relative coordinates
    height = image.shape[0]
    width = image.shape[1]
    for i in range(bboxes.shape[0]):
        xmin = int(np.round(bboxes[i, 1] * width))
        ymin = int(np.round(bboxes[i, 2] * height))
        w = int(np.round(bboxes[i, 3] * width))
        h = int(np.round(bboxes[i, 4] * height))
        classid = int(bboxes[i, 0])
        conf = bboxes[i, 5]
        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, line_width)
        cv2.rectangle(image, (xmin, ymin - 20), (xmin + w, ymin), (125, 125, 125), -1)
        cv2.putText(image, classnames[classid] + ' : %.2f' % conf, (xmin + 5, ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


# ----------------------------------------------------------------------------------------------------------------------
def create_experiment_folder(args):
    year = time.strftime('%Y')
    month = time.strftime('%m')
    day = time.strftime('%d')
    if not os.path.exists(args.experiments_folder):
        os.mkdir(args.experiments_folder)
    year_folder = os.path.join(args.experiments_folder, year)
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)
    base_name = os.path.join(year_folder, year + '_' + month + '_' + day)
    experiment_folder = base_name
    count = 0
    while os.path.exists(experiment_folder):
        count += 1
        experiment_folder = base_name + '_' + str(count)
    os.mkdir(experiment_folder)
    print('Experiment folder: ' + experiment_folder)
    return experiment_folder


# ----------------------------------------------------------------------------------------------------------------------
def copy_config(args, inline_args, configModuleName):
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
                            filename=os.path.join(args.outdir, 'out.log'),
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
        fileh = logging.FileHandler(filename=os.path.join(args.outdir, 'out.log'), mode='w')
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









