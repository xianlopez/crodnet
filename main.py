import argparse
import tensorflow as tf
import numpy as np
import time
import os

# Now TrainConfiguration, EvaluateConfiguration and PredictConfiguration
# are dynamically loaded rather directly loaded. Fot this purpose we use
# importlib package.
import importlib

import tools
import logging
import TrainEnv


# Main function
def main(inline_args):

    ini_time = time.time()

    args = common_stuff(inline_args)

    try:
        if inline_args.run == 'train':
            result = train(args)
        elif inline_args.run == 'evaluate':
            result = evaluate(args, inline_args.split)
        else:
            raise Exception('Please, select specify a valid execution mode: train / evaluate')

        fin_time = time.time()
        print('')
        logging.info('Process finished.')
        logging.info('Total time: %.2f s' % (fin_time - ini_time))

    except Exception as ex:
        logging.error('Fatal error: ' + str(ex))
        raise

    return result


def import_config_files(inline_args):

    configModuleName = 'train_config'
    class2load = "UpdateTrainConfiguration"

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


def common_stuff(inline_args):

    # Set visible GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(inline_args.gpu)

    # Import appropriate config file as interactive module loading:
    currentConfiguration = import_config_files(inline_args)

    # Get arguments from current configuration:
    args = currentConfiguration()

    # Set level of TensorFlow logger:
    if args.tf_log_level == 'SILENT':
        level = 3
    elif args.tf_log_level == 'ERROR':
        level = 2
    elif args.tf_log_level == 'WARNING':
        level = 1
    elif args.tf_log_level == 'INFO':
        level = 0
    else:
        err_msg = 'TensorFlow log level not understood.'
        logging.error(err_msg)
        raise Exception(err_msg)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)

    # Create experiment folder:
    if args.experiments_folder[0] == '.':  # Relative path
        args.experiments_folder = os.path.join(os.getcwd(), args.experiments_folder[2:])
    args.outdir = tools.create_experiment_folder(args)
    # Configure logger:
    tools.configure_logging(args)

    # Copy configuration file to the exeperiment folder:
    try:
        tools.copy_config(args, inline_args)
    except:
        err_msg = 'Error copying config file.'
        logging.error(err_msg)
        raise Exception(err_msg)

    # Set random seed:
    if args.random_seed is not None:
        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Common Representatins for Object Detection')
    parser.add_argument('-r', '--run', type=str, default=None, 
                        help='run mode options: train / evaluate')
    parser.add_argument('-gpu', type=int, default=0,
                        help='GPU ID on which to execute')
    parser.add_argument('-conf', type=str, default=None,
                        help='Choose an existing configuration in .' + os.sep + 'config' + os.sep + ' folder. Ignore the initial ''*_config_'' part. '
                             'If not specified, uses train_config.py or eval_config.py depending on -r argument.' )
    parser.add_argument('-split', type=str, default='val',
                        help='Split on which to evaluate')

    arguments = parser.parse_args()

    assert arguments.run is not None, 'Please, specify run mode: train / evaluate'
    return arguments


def train(args):
    print('run mode: train')

    te = TrainEnv.TrainEnv(args)

    result = te.train()

    return result


def evaluate(args, split):
    print('run mode: evaluate')

    te = TrainEnv.TrainEnv(args)

    result = te.evaluate(split)

    return result


# Entry point of the script
if __name__ == "__main__":
    inline_args = parse_args()
    main(inline_args)