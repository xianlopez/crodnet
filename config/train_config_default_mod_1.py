from config.train_config_base import TrainConfiguration
from DataAugmentation import DataAugOpts
from LRScheduler import LRPolicies, LRSchedulerOpts
import CrodnetOptions
from ImageCropper import ImageCropperOptions


class UpdateTrainConfiguration(TrainConfiguration):

    ##################################
    ######### TRAINING OPTS ##########
    num_epochs = 25
    optimizer_name = 'momentum'  # 'sgd', 'adam', 'rmsprop'
    learning_rate = 1e-3
    #learning_rate = 0
    momentum = 0.9
    # l2_regularization = 5e-4
    l2_regularization = 2e-4
    # l2_regularization = 0
    #nbatches_accum = 2
    train_selected_layers = False
    layers_list = ['vgg_16/conv1/', 'vgg_16/conv2/', 'vgg_16/conv3/', 'vgg_16/conv4/', 'vgg_16/conv5/']
    # layers_list = []
    ##################################

    lr_scheduler_opts = LRSchedulerOpts(LRPolicies.scheduled)
    lr_scheduler_opts.scheduledPolicyOpts.epochsLRDict = {10: 1e-4, 20: 1e-5, 24: 1e-6}
    #lr_scheduler_opts.scheduledPolicyOpts.epochsLRDict = {15: 1e-5}

    single_cell_opts = CrodnetOptions.SingleCellOptions()
    # single_cell_opts.debug_train = True
    # single_cell_opts.debug_eval = True
    single_cell_opts.loc_loss_factor = 1.0
    single_cell_opts.cm_loss_factor = 0
    single_cell_opts.n_comparisons_inter = 0
    single_cell_opts.n_comparisons_intra = 0
    single_cell_opts.threshold_ar_low = 0.05
    single_cell_opts.threshold_ar_low_neutral = 0.04
    single_cell_opts.threshold_ar_high = 0.9
    single_cell_opts.threshold_ar_high_neutral = 0.9
    single_cell_opts.threshold_dc = 0.5
    single_cell_opts.threshold_dc_neutral = 0.7
    single_cell_opts.cm_same_class = True
    single_cell_opts.th_cm_background = 0.15
    single_cell_opts.th_cm_neutral = 0.20
    single_cell_opts.n_images_per_batch = 16
    single_cell_opts.n_crops_per_image = 8

    image_cropper_opts = ImageCropperOptions()
    image_cropper_opts.max_dc = 0.9
    image_cropper_opts.min_ar = 0.02
    image_cropper_opts.max_ar = 1
    image_cropper_opts.probability_focus = 0.45
    image_cropper_opts.probability_pair = 0.3
    image_cropper_opts.probability_inside = 0
    image_cropper_opts.max_dc_pair = 0.5

    #multi_cell_opts = CrodnetOptions.MultiCellOptions()
    #multi_cell_opts.grid_levels_size_pad = [
    #    (256, 64),
    #    (288, 96),
    #    # (352, 64),
    #    (416, 64),
    #    # (512, 64),
    #    (608, 64),
    #    (704, 64),
    #    (832, 64),
    #    (896, 64)
    #]
    #multi_cell_opts.cm_same_class = True
    #multi_cell_opts.n_images_per_batch = 1
    #multi_cell_opts.threshold_ar_low = 0.05
    #multi_cell_opts.threshold_ar_high = 0.9


    #dataset_name = 'VOC0712_filtered'  # Any folder in the <<root_of_datasets>> directory.
    dataset_name = 'VOC0712'  # Any folder in the <<root_of_datasets>> directory.

    debug_hnm = False

    ##################################
    ######### INITIALIZATION #########
    initialization_mode = 'load-pretrained'  # 'load-pretrained', 'scratch'
    weights_file = r'C:\development\crodnet\experiments\2019\ssd_training_2019_01_10\model-240'
    modified_scopes = ['new_layers', 'prediction', 'comparison']
    restore_optimizer = False
    #weights_file = '/home/xian/crodnet/experiments/2019/2019_03_14_2/model-18'
    #modified_scopes = []
    # restore_optimizer = True
    #restore_optimizer = False
    ##################################


    ##################################
    ####### DATA AUGMENTATION ########
    data_aug_opts = DataAugOpts()
    data_aug_opts.apply_data_augmentation = True
    data_aug_opts.horizontal_flip = True
    data_aug_opts.random_brightness = True
    data_aug_opts.random_contrast = True
    data_aug_opts.random_saturation = True
    data_aug_opts.random_hue = True
    data_aug_opts.convert_to_grayscale_prob = 0.05
    # data_aug_opts.write_image_after_data_augmentation = True
    ##################################


    nepochs_save = 1
    nepochs_checkval = 1
    nsteps_display = 20
    nepochs_hnm = 50
    nepochs_mceval = 50


    #num_workers = 3
    #percent_of_data = 0.5
    # random_seed = 1

