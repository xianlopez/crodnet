# ======================================================================================================================
import TrainDataReader
import tensorflow as tf
import tools
import logging
import time
import numpy as np
import operator
import webbrowser
import subprocess
import socket
import os
from L2Regularization import L2RegularizationLoss
import math
import SingleCellArch
import re
from LRScheduler import LRScheduler
import MultiCellEnv
import shutil


class Checkpoint:
    def __init__(self, path, val_loss):
        self.path = path
        self.val_loss = val_loss


# ======================================================================================================================
class TrainEnv:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, opts):

        self.opts = opts

        self.loss = None
        self.optimizer = None
        self.predictions = None
        self.inputs = None
        self.labels = None
        self.filenames = None
        self.train_op = None
        self.saver = None
        self.init_op = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.input_shape = None
        self.model_variables = None
        self.reader = None
        self.tensorboard_process = None

        self.nbatches_accum = self.opts.nbatches_accum
        self.zero_ops = None
        self.accum_ops = None
        self.train_step = None

        self.graph_hnm = tf.Graph()
        with self.graph_hnm.as_default():
            self.env_hnm = MultiCellEnv.MultiCellEnv(self.opts, 'train', HNM=True)

        self.graph_mceval = tf.Graph()
        with self.graph_mceval.as_default():
            self.env_mceval = MultiCellEnv.MultiCellEnv(self.opts, 'val', HNM=False)

        # Initialize network:
        self.graph_train = tf.Graph()
        with self.graph_train.as_default():
            self.generate_graph()

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, split):
        print('')
        logging.info('Start evaluation')
        with tf.Session(graph=self.graph_train, config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            self.initialize(sess)
            loss_mean, metrics_mean = self.evaluate_on_dataset(split, sess)
            logging.info('Loss: %.2e' % loss_mean)
            for m_idx in range(self.single_cell_arch.n_metrics):
                logging.info(self.single_cell_arch.metric_names[m_idx] + ': %.2f' % metrics_mean[m_idx])

        return

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_on_dataset(self, split, sess):
        logging.info('Computing loss and metrics on ' + split + ' data')
        initime = time.time()
        sess.run(self.reader.get_init_op(split))
        nbatches = self.reader.get_nbatches_per_epoch(split)
        step = 0
        loss_mean = 0
        metrics_mean = np.zeros(shape=(self.single_cell_arch.n_metrics))
        while True:
            try:
                batch_loss, batch_metrics = sess.run([self.loss, self.metrics])
                loss_mean = (loss_mean * step + batch_loss) / (step + 1)
                metrics_mean = (metrics_mean * step + batch_metrics) / (step + 1)
                step += 1
            except tf.errors.OutOfRangeError:
                break
            if step % self.opts.nsteps_display == 0:
                print('Step %i / %i, loss: %.2e' % (step, nbatches, batch_loss))
        fintime = time.time()
        logging.debug('Done in %.2f s' % (fintime - initime))

        return loss_mean, metrics_mean

    # ------------------------------------------------------------------------------------------------------------------
    def train(self):

        print('')
        logging.info("Start training")
        assert self.opts.nepochs_mceval % self.opts.nepochs_save == 0, 'nepochs_mceval must be a multiple of nepochs_save'
        assert self.opts.nepochs_hnm % self.opts.nepochs_save == 0, 'nepochs_hnm must be a multiple of nepochs_save'
        nbatches_train = self.reader.get_nbatches_per_epoch('train')
        lr_scheduler = LRScheduler(self.opts.lr_scheduler_opts, self.opts.outdir)
        sess = tf.Session(graph=self.graph_train, config=tools.get_config_proto(self.opts.gpu_memory_fraction))
        try:
            # Initialization:
            self.initialize(sess)
            # Lists for the training history:
            train_metrics_history = []
            train_losses_history = []
            val_metrics_history = []
            val_losses_history = []
            checkpoints = [] # This is a list of Checkpoint objects.

            # Tensorboard:
            with self.graph_train.as_default():
                merged, summary_writer, tensorboard_url = self.prepare_tensorboard(sess)

            # Loop on epochs:
            current_lr = self.opts.learning_rate
            global_step = 0
            for epoch in range(1, self.opts.num_epochs + 1):
                print('')
                logging.info('Starting epoch %d / %d' % (epoch, self.opts.num_epochs))
                sess.run(self.reader.get_init_op('train'))
                current_lr = lr_scheduler.GetLearningRateAtEpoch(epoch, current_lr)
                _ = sess.run([self.update_lr_op], feed_dict={self.lr_to_update: current_lr})
                learning_rate = sess.run([self.learning_rate])[0]
                logging.info('Learning rate: ' + str(learning_rate))
                step = 0
                loss_mean = 0
                metrics_mean = np.zeros(shape=(self.single_cell_arch.n_metrics))
                iniepoch = time.time()
                while True:
                    try:
                        if self.nbatches_accum > 0:
                            if step % self.nbatches_accum == 0:
                                sess.run(self.zero_ops)
                            _, batch_loss, batch_metrics, summaryOut = sess.run([self.accum_ops, self.loss, self.metrics, merged])
                            if (step + 1) % self.nbatches_accum == 0:
                                _ = sess.run([self.train_step])
                        else:
                            _, batch_loss, batch_metrics, summaryOut = sess.run([self.train_op, self.loss, self.metrics, merged])

                        if math.isnan(batch_loss):
                            raise Exception("Loss is Not A Number")

                        loss_mean = (loss_mean * step + batch_loss) / (step + 1)
                        metrics_mean = (metrics_mean * step + batch_metrics) / (step + 1)
                        step += 1
                        global_step += 1

                        # Tensorboard:
                        summary_writer.add_summary(summaryOut, global_step)
                        if global_step == 1:
                            webbrowser.open_new_tab(tensorboard_url)

                    except tf.errors.OutOfRangeError:
                        break

                    if step % self.opts.nsteps_display == 0:
                        logging.info('Step %i / %i, loss: %.2e' % (step, nbatches_train, batch_loss))

                finepoch = time.time()
                logging.debug('Epoch computed in %.2f s' % (finepoch - iniepoch))

                # Distribution of crops:
                pcnt_obj_focus, pcnt_hn_focus, pcnt_pos_crops, pcnt_neutral_crops, pcnt_neg_crops = self.reader.compute_and_reset_statistics()
                logging.info('Percent of crops focusing on a positive object: %.2f' % (pcnt_obj_focus))
                logging.info('Percent of crops focusing on a hard negative: %.2f' % (pcnt_hn_focus))
                logging.info('Percent of positive crops: %.2f' % (pcnt_pos_crops))
                logging.info('Percent of neutral crops: %.2f' % (pcnt_neutral_crops))
                logging.info('Percent of negative crops: %.2f' % (pcnt_neg_crops))

                # Compute loss and metrics on training data:
                if self.opts.recompute_train:
                    train_loss, train_metrics = self.evaluate_on_dataset('train', sess)
                    logging.info('Train loss: %.2e' % train_loss)
                    for m_idx in range(self.single_cell_arch.n_metrics):
                        logging.info('Train ' + self.single_cell_arch.metric_names[m_idx] + ': %.2f' % train_metrics_history[m_idx])
                else:
                    train_loss = loss_mean
                    train_metrics = metrics_mean
                    logging.info('Mean train loss during epoch: %.2e' % train_loss)
                    for m_idx in range(self.single_cell_arch.n_metrics):
                        logging.info('Mean train ' + self.single_cell_arch.metric_names[m_idx] + ' during epoch: %.2f' % train_metrics[m_idx])
                train_losses_history.append(train_loss)
                train_metrics_history.append(train_metrics)

                # Compute loss and metrics on validation data:
                if epoch % self.opts.nepochs_checkval == 0:
                    val_loss, val_metrics = self.evaluate_on_dataset('val', sess)
                    val_losses_history.append(val_loss)
                    val_metrics_history.append(val_metrics)
                    logging.info('Val loss: %.2e' % val_loss)
                    for m_idx in range(self.single_cell_arch.n_metrics):
                        logging.info('Val ' + self.single_cell_arch.metric_names[m_idx] + ': %.2f' % val_metrics[m_idx])
                else:
                    val_loss = None
                    val_metrics = None

                # Plot training progress:
                tools.plot_training_history(train_metrics_history, train_losses_history,
                                            val_metrics_history, val_losses_history,
                                            self.single_cell_arch.metric_names, self.opts, epoch)

                # Save the model:
                if epoch % self.opts.nepochs_save == 0:
                    self.save_checkpoint(sess, val_loss, epoch, checkpoints, self.opts.outdir)

                # Hard-negative mining:
                if epoch % self.opts.nepochs_hnm == 0:
                    sess.close()
                    self.clean_hard_negatives()
                    logging.info('Looking for hard negatives...')
                    with self.graph_hnm.as_default():
                        train_mAP = self.env_hnm.evaluate_and_hnm(tf.train.latest_checkpoint(self.opts.outdir))
                    if epoch % self.opts.nepochs_mceval != 0:
                        logging.info('Starting training session again.')
                        sess = tf.Session(graph=self.graph_train, config=tools.get_config_proto(self.opts.gpu_memory_fraction))
                        self.saver.restore(sess, tf.train.latest_checkpoint(self.opts.outdir))
                else:
                    train_mAP = None

                # Check mAP in validation data:
                if epoch % self.opts.nepochs_mceval == 0:
                    if epoch % self.opts.nepochs_hnm != 0:
                        sess.close()
                    logging.info('Computing mAP on validation data...')
                    with self.graph_mceval.as_default():
                        val_mAP = self.env_mceval.evaluate(tf.train.latest_checkpoint(self.opts.outdir))
                    logging.info('Starting training session again.')
                    sess = tf.Session(graph=self.graph_train, config=tools.get_config_proto(self.opts.gpu_memory_fraction))
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.opts.outdir))
                else:
                    val_mAP = None

                # Save epoch report:
                self.write_train_report(epoch, train_loss, train_metrics, train_mAP, val_loss, val_metrics, val_mAP)

            # Save the model (if we haven't done it yet):
            if self.opts.num_epochs % self.opts.nepochs_save != 0:
                self.save_checkpoint(sess, val_loss, epoch, checkpoints, self.opts.outdir)

            sess.close()

        except:
            logging.error('Some error happened. Closing session.')
            sess.close()
            raise

        self.end_tensorboard()

        return

    def write_train_report(self, epoch, train_loss, train_metrics, train_mAP, val_loss, val_metrics, val_mAP):
        report_file = os.path.join(self.opts.outdir, 'train_report.csv')
        # The first time, create the file and write headers:
        if not os.path.exists(report_file):
            with open(report_file, 'w') as fid:
                fid.write('Epoch,Train loss,')
                for metric_name in self.single_cell_arch.metric_names:
                    fid.write('Train ' + metric_name + ',')
                fid.write('Train mAP,')
                fid.write('Val loss,')
                for metric_name in self.single_cell_arch.metric_names:
                    fid.write('Val ' + metric_name + ',')
                fid.write('Val mAP\n')
        # Add this epoch data:
        with open(report_file, 'a') as fid:
            fid.write(str(epoch) + ',' + str(train_loss) + ',')
            for i in range(len(self.single_cell_arch.metric_names)):
                fid.write(str(train_metrics[i]) + ',')
            if train_mAP is None:
                fid.write('-,')
            else:
                fid.write(str(train_mAP) + ',')
            if val_loss is None:
                fid.write('-,')
            else:
                fid.write(str(val_loss) + ',')
            if val_metrics is None:
                for i in range(len(self.single_cell_arch.metric_names)):
                    fid.write('-,')
            else:
                for i in range(len(self.single_cell_arch.metric_names)):
                    fid.write(str(val_metrics[i]) + ',')
            if val_mAP is None:
                fid.write('-\n')
            else:
                fid.write(str(val_mAP) + '\n')

    def clean_hard_negatives(self):
        hn_dir = os.path.join(self.opts.outdir, 'hard_negatives')
        if os.path.exists(hn_dir):
            shutil.rmtree(hn_dir)
        hn_imgs_dir = os.path.join(self.opts.outdir, 'hard_negatives_images')
        if os.path.exists(hn_imgs_dir):
            shutil.rmtree(hn_imgs_dir)

    def create_links_to_current_ckpt(self, checkpoints):
        assert len(checkpoints) > 0, 'Trying to create link to current checkpoint, but len(checkpoints) == 0.'
        last_ckpt = checkpoints[len(checkpoints) - 1]
        data_file = None
        index_file = None
        meta_file = None
        data_suffix = None
        for file in os.listdir(self.opts.outdir):
            file_path = os.path.join(self.opts.outdir, file)
            if last_ckpt in file_path:
                if '.data-' in file_path:
                    data_file = file_path
                    pos = file_path.find('.data-')
                    data_suffix = file_path[pos:]
                elif '.index' in file_path:
                    index_file = file_path
                elif '.meta' in file_path:
                    meta_file = file_path
        assert data_file is not None, 'data_file is None'
        assert index_file is not None, 'index_file is None'
        assert meta_file is not None, 'meta_file is None'
        print('data_file = ' + str(data_file))
        print('index_file = ' + str(index_file))
        print('meta_file = ' + str(meta_file))
        data_lnk = os.path.join(self.opts.outdir, 'current_ckpt' + data_suffix)
        index_lnk = os.path.join(self.opts.outdir, 'current_ckpt.index')
        meta_lnk = os.path.join(self.opts.outdir, 'current_ckpt.meta')
        os.symlink(data_file, data_lnk)
        os.symlink(index_file, index_lnk)
        os.symlink(meta_file, meta_lnk)

    def save_checkpoint(self, sess, validation_loss, epoch, checkpoints, outdir):
        if validation_loss is None:
            validation_loss = -1
        # Save new model:
        save_path = self.saver.save(sess, os.path.join(outdir, 'model'), global_step=epoch)
        logging.info('Model saved to ' + save_path)
        new_checkpoint = Checkpoint(save_path, validation_loss)

        # Get the best loss among all the checkpoints (including the new one):
        best_loss = validation_loss
        for ckpt in checkpoints:
            best_loss = min(best_loss, ckpt.val_loss)

        if len(checkpoints) > 0:
            # Remove all the previous checkpoints but the best one.
            checkpoints.sort(key=operator.attrgetter('val_loss'), reverse=True)
            for i in range(len(checkpoints) - 2, -1, -1):
                # Remove:
                ckpt = checkpoints[i]
                folder, name = os.path.split(ckpt.path)
                for file in os.listdir(folder):
                    if re.search(name, file) is not None:
                        file_path = os.path.join(folder, file)
                        try:
                            os.remove(file_path)
                        except Exception as ex:
                            logging.warning('Error deleting file ' + file_path, exc_info=ex)
                checkpoints.pop(i)
                logging.info('Deleted checkpoint. Path: ' + ckpt.path + '  -  Val loss: ' + str(ckpt.val_loss))
            # If the remeaining checkpoint is worse than the new checkpoint, remove it too.
            ckpt = checkpoints[0]
            if ckpt.val_loss >= validation_loss:
                # Remove:
                folder, name = os.path.split(ckpt.path)
                for file in os.listdir(folder):
                    if re.search(name, file) is not None:
                        file_path = os.path.join(folder, file)
                        try:
                            os.remove(file_path)
                        except Exception as ex:
                            logging.warning('Error deleting file ' + file_path, exc_info=ex)
                checkpoints.pop(0)
                logging.info('Deleted checkpoint. Path: ' + ckpt.path + '  -  Val loss: ' + str(ckpt.val_loss))

        # Append the new checkpoint to the list:
        checkpoints.append(new_checkpoint)

        logging.info('Remaining checkpoints:')
        for ckpt in checkpoints:
            logging.info('Path: ' + ckpt.path + '  -  Val loss: ' + str(ckpt.val_loss))
        return checkpoints

    # ------------------------------------------------------------------------------------------------------------------
    def generate_graph(self):
        dirdata = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name)
        img_extension, self.classnames = tools.process_dataset_config(os.path.join(dirdata, 'dataset_info.xml'))
        self.nclasses = len(self.classnames)
        self.single_cell_arch = SingleCellArch.SingleCellArch(self.opts.single_cell_opts, self.nclasses, self.opts.outdir)
        self.define_inputs_and_labels()
        _, self.loss, self.metrics = self.single_cell_arch.make(self.inputs, self.labels, self.filenames)
        self.model_variables = [n.name for n in tf.global_variables()]
        if self.opts.l2_regularization > 0:
            self.loss += L2RegularizationLoss(self.opts)
        self.loss = tf.identity(self.loss, name='loss') # This is just a workaround to rename the loss function to 'loss'
        # Tensorboard:
        tf.summary.scalar("final_loss", self.loss)

        self.build_optimizer()

        self.define_initializer()
        self.saver = tf.train.Saver(name='net_saver', max_to_keep=1000000)

    # ------------------------------------------------------------------------------------------------------------------
    def define_inputs_and_labels(self):
        self.input_shape = self.single_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = TrainDataReader.TrainDataReader(self.input_shape, self.opts, self.single_cell_arch)
            self.inputs, self.labels, self.filenames = self.reader.build_iterator()
        self.single_cell_arch.set_classnames(self.classnames)

    # ------------------------------------------------------------------------------------------------------------------
    def define_initializer(self):

        if self.opts.initialization_mode == 'load-pretrained':

            # self.model_variables has all the model variables (it doesn't include the optimizer variables
            # or others).
            # We will filter out from it the variables that fit in the scopes specified in opts.modified_scopes.
            # Usually this is necessary in the last layer, if the number of outputs is different.
            assert type(self.opts.modified_scopes) == list, 'modified_scopes should be a list.'
            varnames_to_restore = []

            print('')
            logging.debug('Variables to restore:')
            candidate_vars_to_restore = self.model_variables
            if self.opts.restore_optimizer:
                candidate_vars_to_restore.extend([var.name for var in self.optimizer_variables])
            for var in candidate_vars_to_restore:
                is_modified = False
                for modscope in self.opts.modified_scopes:
                    if modscope in var:
                        is_modified = True
                if not is_modified:
                    varnames_to_restore.append(var)
                    logging.debug(var)

            # Get the variables to restore:
            vars_to_restore = tf.contrib.framework.get_variables_to_restore(include=varnames_to_restore)
            self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, vars_to_restore)
            self.ckpt_restore = tf.placeholder(shape=(), dtype=tf.string)

            # Variables to initialize from scratch (the rest):
            vars_new = tf.contrib.framework.get_variables_to_restore(exclude=varnames_to_restore)
            self.init_op = tf.variables_initializer(vars_new, name='scratch_vars_init')

        elif self.opts.initialization_mode == 'scratch':
            self.init_op = tf.global_variables_initializer()

        else:
            raise Exception('Initialization mode not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def initialize(self, sess):

        if self.opts.initialization_mode == 'load-pretrained':
            self.restore_fn(sess)
            sess.run(self.init_op)
        elif self.opts.initialization_mode == 'scratch':
            sess.run(self.init_op)
        else:
            raise Exception('Initialization mode not recognized.')

    # ------------------------------------------------------------------------------------------------------------------
    def build_optimizer(self):

        # Choose the variables to train:
        vars_to_train = tools.get_trainable_variables(self.opts)

        print('')
        logging.debug('Training variables:')

        for v in vars_to_train:
            logging.debug(v.name)

        # if self.nbatches_accum > 0:
        #     opts.learning_rate = opts.learning_rate / self.nbatches_accum

        self.learning_rate = tf.Variable(initial_value=self.opts.learning_rate, dtype=tf.float32, name='learning_rate')
        self.lr_to_update = tf.placeholder(dtype=tf.float32, shape=(), name='lr_to_update')
        self.update_lr_op = tf.assign(self.learning_rate, self.lr_to_update, name='UpdateLR')

        previous_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Decide what optimizer to use:
        if self.opts.optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.opts.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opts.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.opts.optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.opts.momentum)
        else:
            raise Exception('Optimizer name not recognized.')

        if self.nbatches_accum > 0:
            with tf.name_scope('n_batches_accum'):
                # assert opts.optimizer_name == 'sgd', 'If nbatches_accum > 0, the optimizer must be SGD'
                accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value(), name='zeros_tv'), trainable=False, name='accum_grads') for tv in vars_to_train]
                self.zero_ops = [ag.assign(tf.zeros_like(ag, name='zeros_ag'), name='zero_ops_assign') for ag in accum_grads]
                gvs = optimizer.compute_gradients(self.loss, vars_to_train)
                self.accum_ops = [accum_grads[i].assign_add(gv[0] / float(self.nbatches_accum), name='assign_add_gv') for i, gv in enumerate(gvs)]
                self.train_step = optimizer.apply_gradients([(accum_grads[i], gv[1]) for i, gv in enumerate(gvs)], name='ntrain_step')

        else:
            self.train_op = optimizer.minimize(self.loss, var_list=vars_to_train, name='train_op')

        posterior_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.optimizer_variables = []
        for var in posterior_variables:
            if var not in previous_variables:
                self.optimizer_variables.append(var)

        return


    # ----------------------------------------------------------------------------------------------------------------------
    def end_tensorboard(self):
        print('end_tensorboard')
        print(self.tensorboard_process)
        print('pid: ' + str(self.tensorboard_process.pid))
        if self.tensorboard_process is not None:
            logging.info('Killing tensorboard')
            self.tensorboard_process.kill()
        return


    # ----------------------------------------------------------------------------------------------------------------------
    def prepare_tensorboard(self, sess):
        merged = tf.summary.merge_all(name='summary_merge_all')
        summary_writer = tf.summary.FileWriter(os.path.join(self.opts.outdir, 'tensorboard'), sess.graph)
        if os.name == 'nt':  # Windows
            command = r'C:\development\venvs\brainlab_tf11\Scripts\tensorboard --logdir=' + os.path.join(self.opts.outdir, 'tensorboard')
            self.tensorboard_process = subprocess.Popen(["start", "cmd", "/k", command], shell=True)
        elif os.name == 'posix':  # Ubuntu
            command = 'python /home/xian/venvs/brainlab_tf11/lib/python3.5/site-packages/tensorboard/main.py --logdir=' + \
                      os.path.join(self.opts.outdir, 'tensorboard')
            self.tensorboard_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        else:
            raise Exception('Operative system name not recognized: ' + str(os.name))
        hostname = socket.gethostname()
        tensorboard_url = 'http://' + hostname + ':6006'
        return merged, summary_writer, tensorboard_url