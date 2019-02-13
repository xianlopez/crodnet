from DataAugmentation import DataAugOpts
import CrodnetOptions
from LRScheduler import LRPolicies, LRSchedulerOpts
from ImageCropper import ImageCropperOptions
import os


########### ALL CONFIG ###########
class EvaluateConfiguration:

    dataset_name = ''  # Any folder in the <<root_of_datasets>> directory.

    weights_file = r''

    lr_scheduler_opts = LRSchedulerOpts(LRPolicies.onCommand)

    data_aug_opts = DataAugOpts()

    crodnet_opts = CrodnetOptions.CrodnetOptions()

    single_cell_opts = CrodnetOptions.SingleCellOptions()

    image_cropper_opts = ImageCropperOptions()

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
    write_network_input = False
    shuffle_data = True
    ##################################


    ##################################
    ##################################
    # The following code should not be touched:
    outdir = None
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    modified_scopes = []
    restore_optimizer = False
