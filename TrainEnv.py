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

        # Initialize network:
        self.generate_graph()

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
        nbatches_train = self.reader.get_nbatches_per_epoch('train')
        lr_scheduler = LRScheduler(self.opts.lr_scheduler_opts, self.opts.outdir)
        with tf.Session(config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            # Initialization:
            self.initialize(sess)
            # Lists for the training history:
            train_metrics = []
            train_losses = []
            val_metrics = []
            val_losses = []
            val_loss = None
            checkpoints = [] # This is a list of Checkpoint objects.

            # Tensorboard:
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

                # Compute loss and metrics on training data:
                if epoch % self.opts.nepochs_checktrain == 0:
                    if self.opts.recompute_train:
                        loss_mean, metrics_mean = self.evaluate_on_dataset('train', sess)
                        logging.info('Train loss: %.2e' % loss_mean)
                        for m_idx in range(self.single_cell_arch.n_metrics):
                            logging.info('Train ' + self.single_cell_arch.metric_names[m_idx] + ': %.2f' % train_metrics[m_idx])
                    else:
                        logging.info('Mean train loss during epoch: %.2e' % loss_mean)
                        for m_idx in range(self.single_cell_arch.n_metrics):
                            logging.info('Mean train ' + self.single_cell_arch.metric_names[m_idx] + ' during epoch: %.2f' % metrics_mean[m_idx])
                    train_losses.append(loss_mean)
                    train_metrics.append(metrics_mean)

                # Compute loss and metrics on validation data:
                if epoch % self.opts.nepochs_checkval == 0:
                    val_loss, metrics = self.evaluate_on_dataset('val', sess)
                    val_losses.append(val_loss)
                    val_metrics.append(metrics)
                    logging.info('Val loss: %.2e' % val_loss)
                    for m_idx in range(self.single_cell_arch.n_metrics):
                        logging.info('Val ' + self.single_cell_arch.metric_names[m_idx] + ': %.2f' % metrics[m_idx])
                else:
                    val_loss = None

                # Plot training progress:
                if epoch % self.opts.nepochs_checktrain == 0 or epoch % self.opts.nepochs_checkval == 0:
                    tools.plot_training_history(train_metrics, train_losses, val_metrics, val_losses, self.single_cell_arch.metric_names, self.opts, epoch)

                # Save the model:
                if epoch % self.opts.nepochs_save == 0:
                    self.save_checkpoint(sess, val_loss, epoch, checkpoints, self.opts.outdir)

            # Save the model (if we haven't done it yet):
            if self.opts.num_epochs % self.opts.nepochs_save != 0:
                self.save_checkpoint(sess, val_loss, epoch, checkpoints, self.opts.outdir)

        self.end_tensorboard()

        return

    def save_checkpoint(self, sess, validation_loss, epoch, checkpoints, outdir):
        if validation_loss is None:
            validation_loss = -1
        # Save new model:
        save_path = self.saver.save(sess, tools.join_paths(outdir, 'model'), global_step=epoch)
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
        self.single_cell_arch = SingleCellArch.SingleCellArch(self.opts.single_cell_opts, TrainDataReader.get_n_classes(self.opts), self.opts.outdir)
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
        self.saver = tf.train.Saver(name='net_saver')

    # ------------------------------------------------------------------------------------------------------------------
    def define_inputs_and_labels(self):
        self.input_shape = self.single_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = TrainDataReader.TrainDataReader(self.input_shape, self.opts, self.single_cell_arch)
            self.inputs, self.labels, self.filenames = self.reader.build_iterator()
        self.classnames = self.reader.classnames
        print('class names before setting')
        print(self.classnames)
        self.single_cell_arch.set_classnames(self.classnames)
        print('class names after setting')
        print(self.classnames)
        self.nclasses = len(self.classnames)

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