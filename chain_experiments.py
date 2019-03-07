import tensorflow as tf
from SingleCellMain import main
import numpy as np

class InlineArgs:
    gpu = '0'
    run = 'train'
    conf = ''

# The file chain_config_files.txt must contain one line with the name of each config file that you want
# to execute. This file names should not contain the part 'train_config_' at the beggining. For instance,
# if you want to include the config file named 'train_config_chain_1.py', you should add a line in the file
# chain_config_files.txt with the text 'chain_1'.
with open('chain_config_files.txt', 'r') as fid:
    config_files = fid.read().split('\n')
    config_files = [file_name for file_name in config_files if file_name != '']  # Remove empty lines.

n_experiments = len(config_files)
results = -np.ones(shape=(n_experiments), dtype=np.float32)

for i in range(n_experiments):
    config = config_files[i]
    try:
        print('==============================================================================================')
        print('==============================================================================================')
        print('Experiment: ' + config)
        inline_args = InlineArgs()
        inline_args.conf = config
        tf.reset_default_graph()
        results[i] = main(inline_args)
    except:
        print('Exception in experiment. Going to the next one')

    print('')
    print('==============================================================================================')
    print('Partial results:')
    for j in range(i + 1):
        print(config_files[j] + ': ' + str(results[j]))

print('')
print('==============================================================================================')
print('==============================================================================================')
print('All experiments done.')
print('Results:')
for i in range(n_experiments):
    print(config_files[i] + ': ' + str(results[i]))