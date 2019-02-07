from mean_ap import MeanAPOpts
from crodnet import crodnet_options
from DataAugmentation import DataAugOpts


class EvaluateConfiguration:
    batch_size = 32

    model_name = ''  # 'vgg16', 'mnistnet', 'yolo'
    dataset_name = ''  # Any folder in the <<root_of_datasets>> directory.
    split = 'val'  # Subset on which to perform the evaluation ('train', 'val')

    write_images = True
    write_results = True

    # Weights initialization:
    # To start from sratch, choose 'scratch'
    # To load pretrained weights, and start training with them, choose 'load-pretrained'
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    # To load pretrained weights:
    weights_file = r''
    modified_scopes = []


    ##################################
    ############ RESIZING ############
    # Select the way to fit the image to the size required by the network.
    # For DETECTION, use ONLY RESIZE_WARP.
    # 'resize_warp': Resize both sides of the image to the required sizes. Aspect ratio may be changed.
    # 'resize_pad_zeros': Scale the image until it totally fits inside the required shape. We pad with zeros the areas
    #                     in which there is no image. Aspect ratio is preserved.
    # 'resize_lose_part': Scale the image until it totally covers the area that the network will see. We may lose the
    #                     upper and lower parts, or the left and right ones. Aspect ratio is preserved.
    # 'centered_crop': Take a centered crop of the image. If any dimension of the image is smaller than the input
    #                  size, we pad with zeros.
    resize_method = 'resize_warp'  # 'resize_warp', 'resize_pad_zeros', 'resize_lose_part', 'centered_crop'
    ##################################


    mean_ap_opts = MeanAPOpts()

    crodnet_opts = crodnet_options()


    ##################################
    ####### ONLY FOR DETECTION #######
    th_conf_detection_evaluate = 0.1
    th_conf_detection_predict = 0.5
    # grid_size = 7  # The amount of horizontal (and vertical) cells in which we will divide the image
    # threshold = 0.2  # Confidence threshold
    threshold_iou_map = 0.5  # Threshold for intersection over union.
    nonmaxsup = True  # Non-maximum supression
    threshold_nms = 0.5  # Non-maximum supression threshold


    ##################################
    ########### OTHER OPTS ###########
    percent_of_data = 100  # For debbuging. Percentage of data to use. Put 100 if not debbuging
    num_workers = 4  # Number of parallel processes to read the data.
    root_of_datasets = r'D:\datasets'
    experiments_folder = r'.\experiments'
    random_seed = 1234  # An integer number, or None in order not to set the random seed.
    tf_log_level = 'ERROR'
    nsteps_display = 20
    buffer_size = 1000
    save_input_images = False  # For debugging, save the input images to see exactly what the network is receiving after
                              # preprocessing, data augmentation, and resizing. Only works in evaluate mode.
    max_image_size = 700
    gpu_memory_fraction = -1.0
    write_network_input = False
    shuffle_data = True
    ##################################


    # Create structure for data augmentation, but actually do not use any data augmentation.
    data_aug_opts = DataAugOpts()

    nbatches_accum = 0 # 0 to not applyl batch accumulation.
    restore_optimizer = False
    l2_regularization = 0
