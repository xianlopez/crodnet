from DataAugmentation import DataAugOpts
import CrodnetOptions
from LRScheduler import LRPolicies, LRSchedulerOpts
from ImageCropper import ImageCropperOptions
import os
from mean_ap import MeanAPOpts


########### ALL CONFIG ###########
class TrainConfiguration:
    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 5
    optimizer_name = 'sgd'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    momentum = 0.9
    l2_regularization = 5e-4
    vars_to_skip_l2_reg = ['scale', 'bias', 'BatchNorm'] # List with strings contained by the variables that you don't want to add to the L2 regularization loss.
    nbatches_accum = 0 # 0 to not applyl batch accumulation.
    # If train_selected_layers is true, the layers in layers_list are the only ones that are going to be trained.
    # Otherwise, those are the only layers excluded for training.
    # The elements of layers_list do not need to match exactly the layers names. It is enough if they are contained
    # in the layer name. For instance, if we make layers_list = ['fc'] in vgg16, it will include layers fc6, fc7, fc8.
    train_selected_layers = True
    # layers_list = ['fc']  # If this is empy or none, all variables are trained.
    layers_list = []  # If this is empy or none, all variables are trained.
    ##################################

    dataset_name = ''  # Any folder in the <<root_of_datasets>> directory.

    ##################################
    ######### INITIALIZATION #########
    # Weights initialization:
    # To start from sratch, choose 'scratch'
    # To load pretrained weights, and start training with them, choose 'load-pretrained'
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = r''
    modified_scopes = []
    restore_optimizer = False
    ##################################

    lr_scheduler_opts = LRSchedulerOpts(LRPolicies.onCommand)

    data_aug_opts = DataAugOpts()

    single_cell_opts = CrodnetOptions.SingleCellOptions()

    image_cropper_opts = ImageCropperOptions()

    multi_cell_opts = CrodnetOptions.MultiCellOptions()

    mean_ap_opts = MeanAPOpts()

    hard_negatives_factor = 0.25
    detect_against_background = False
    th_conf = 0.8
    th_conf_eval = 0.1
    write_results = False
    threshold_nms = 0.5
    threshold_pcs = 0.6
    threshold_iou = 0.5

    ##################################
    ######## DISPLAYING OPTS #########
    # If recompute_train is false, the metrics and loss shown for a training epoch, are computed with the results
    # obtained with the training batches (thus, not reflecting the performance at the end of the epoch, but during it).
    # Otherwise, we go through all the training data again to compute its loss and metrics. This is more time consuming.
    recompute_train = False
    nsteps_display = 20
    nepochs_save = 100
    nepochs_checktrain = 1
    nepochs_checkval = 1
    nepochs_hnm = 1
    ##################################

    ##################################
    ########### OTHER OPTS ###########
    percent_of_data = 100  # For debbuging. Percentage of data to use. Put 100 if not debbuging
    num_workers = 8  # Number of parallel processes to read the data.
    if os.name == 'nt':  # Windows
        root_of_datasets = r'D:\datasets'
        experiments_folder = r'.\experiments'
    elif os.name == 'posix':  # Linux
        root_of_datasets = r'/home/xian/datasets'
        experiments_folder = r'./experiments'
    else:
        raise Exception('Unexpected OS')
    random_seed = None  # An integer number, or None in order not to set the random seed.
    tf_log_level = 'ERROR'
    buffer_size = 1000 # For shuffling data.
    max_image_size = 600
    gpu_memory_fraction = -1.0
    shuffle_data = True
    ##################################


    ##################################
    ##################################
    # The following code should not be touched:
    outdir = None
