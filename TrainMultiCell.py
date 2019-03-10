import argparse
import time
import os
import tools
import logging
import MultiCellTrainEnv


# Main function
def main(inline_args):

    ini_time = time.time()

    opts = tools.common_stuff(inline_args, 'mct_config', 'UpdateTrainConfiguration')

    try:
        if inline_args.run == 'train':
            train(opts)
        elif inline_args.run == 'evaluate':
            evaluate(opts, inline_args.split)
        else:
            raise Exception('Please, select specify a valid execution mode: train / evaluate')
        fin_time = time.time()
        print('')
        logging.info('Process finished.')
        logging.info('Total time: %.2f s' % (fin_time - ini_time))

    except Exception as ex:
        logging.error('Fatal error: ' + str(ex))
        raise

    return


def train(args):
    print('run mode: train')

    env = MultiCellTrainEnv.MultiCellTrainEnv(args)

    result = env.train()

    return result


def evaluate(args, split):
    print('run mode: evaluate')

    env = MultiCellTrainEnv.MultiCellTrainEnv(args)

    result = env.evaluate(split)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Common Representatins for Object Detection')
    parser.add_argument('-r', '--run', type=str, default=None,
                        help='run mode options: train / evaluate')
    parser.add_argument('-gpu', type=int, default=0,
                        help='GPU ID on which to execute')
    parser.add_argument('-conf', type=str, default=None,
                        help='Choose an existing configuration in .' + os.sep + 'config' + os.sep + ' folder. Ignore the initial ''*_config_'' part. '
                             'If not specified, uses train_config.py depending on -r argument.' )
    parser.add_argument('-split', type=str, default='val',
                        help='Split on which to evaluate')

    arguments = parser.parse_args()

    return arguments


# Entry point of the script
if __name__ == "__main__":
    inline_args = parse_args()
    main(inline_args)