import CrodnetOptions
import os
from mean_ap import MeanAPOpts


class MultiCellConfiguration:

    dataset_name = ''  # Any folder in the <<root_of_datasets>> directory.

    weights_file = r''

    multi_cell_opts = CrodnetOptions.MultiCellOptions()

    mean_ap_opts = MeanAPOpts()

    percent_of_data = 100  # For debbuging. Percentage of data to use. Put 100 if not debbuging
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
    max_image_size = 600
    gpu_memory_fraction = -1.0
    shuffle_data = True
    nsteps_display = 20

    write_results = False

    threshold_nms = 0.5
    threshold_iou = 0.5


    ##################################
    ##################################
    # The following code should not be touched:
    outdir = None
