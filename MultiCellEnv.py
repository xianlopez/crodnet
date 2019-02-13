# ======================================================================================================================
import TrainDataReader
import tensorflow as tf
import tools
import logging
import time
import numpy as np
import MultiCellArch


# ======================================================================================================================
class TrainEnv:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, opts):

        self.opts = opts

        self.labels = None
        self.filenames = None
        self.inputs = None
        self.saver = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.input_shape = None
        self.model_variables = None
        self.reader = None

        # Initialize network:
        self.generate_graph()

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, split):
        print('')
        logging.info('Start evaluation')
        with tf.Session(config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            self.restore_fn(sess)
            logging.info('Computing metrics on ' + split + ' data')
            initime = time.time()
            sess.run(self.reader.get_init_op(split))
            nbatches = self.reader.get_nbatches_per_epoch(split)
            all_predictions = []
            step = 0
            while True:
                try:
                    net_output, common_representations, labels_enc, filenames = \
                        sess.run([self.net_output, self.common_representations, self.labels_enc, self.filenames])
                    batch_predictions = self.postprocess_predictions(net_output, common_representations)
                    all_predictions.extend(batch_predictions)
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
                if step % self.opts.nsteps_display == 0:
                    print('Step %i / %i' % (step, nbatches))
            fintime = time.time()
            logging.debug('Done in %.2f s' % (fintime - initime))

        return

    def postprocess_predictions(self, net_output, common_representations):
        return predictions

    # ------------------------------------------------------------------------------------------------------------------
    def generate_graph(self):
        self.multi_cell_arch = MultiCellArch.MultiCellArch(self.opts.single_cell_opts, TrainDataReader.get_n_classes(self.opts), self.opts.outdir)
        self.define_inputs_and_labels()
        self.net_output, self.common_representations = self.multi_cell_arch.make(self.inputs)
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, tf.global_variables())
        self.saver = tf.train.Saver(name='net_saver', max_to_keep=1000000)

    # ------------------------------------------------------------------------------------------------------------------
    def define_inputs_and_labels(self):
        self.input_shape = self.multi_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = TrainDataReader.TrainDataReader(self.input_shape, self.opts, self.multi_cell_arch)
            self.inputs, self.labels_enc, self.filenames = self.reader.build_iterator()
        self.classnames = self.reader.classnames
        print('class names before setting')
        print(self.classnames)
        self.multi_cell_arch.set_classnames(self.classnames)
        print('class names after setting')
        print(self.classnames)
        self.nclasses = len(self.classnames)


