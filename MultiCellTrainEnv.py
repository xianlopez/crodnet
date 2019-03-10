# ======================================================================================================================
import MultiCellDataReader
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
import cv2
import MultiCellArch
import re
from LRScheduler import LRScheduler
import mean_ap
import BoundingBoxes


class Checkpoint:
    def __init__(self, path, val_loss):
        self.path = path
        self.val_loss = val_loss


# ======================================================================================================================
class MultiCellTrainEnv:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, opts):

        self.opts = opts

        self.loss = None
        self.optimizer = None
        self.predictions = None
        self.inputs = None
        self.labels_enc = None
        self.filenames = None
        self.train_op = None
        self.saver = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.model_variables = None
        self.tensorboard_process = None

        self.nbatches_accum = self.opts.nbatches_accum
        self.zero_ops = None
        self.accum_ops = None
        self.train_step = None

        # Initialize network:
        self.generate_graph()

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, split):
        print('')
        logging.info('Start evaluation')
        with tf.Session(config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            self.initialize(sess)
            loss_mean, metrics_mean = self.evaluate_on_dataset(split, sess)
            logging.info('Loss: %.2e' % loss_mean)
            for m_idx in range(self.arch.n_metrics):
                logging.info(self.arch.metric_names[m_idx] + ': %.2f' % metrics_mean[m_idx])

        return

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_on_dataset(self, split, sess):
        logging.info('Computing loss and metrics on ' + split + ' data')
        reader = self.get_split_reader(split)
        initime = time.time()
        step = 0
        all_gt_boxes = []
        all_pred_boxes = []
        loss_mean = 0
        metrics_mean = np.zeros(shape=(self.arch.n_metrics))
        if self.opts.write_results:
            images_dir = os.path.join(self.opts.outdir, 'images')
            os.makedirs(images_dir)

        end_of_epoch = False
        while not end_of_epoch:
            batch_images, batch_bboxes, batch_filenames, end_of_epoch = self.reader_train.get_next_batch(apply_data_aug=False)
            labels_enc = self.arch.encode_gt_batched(batch_bboxes)
            localizations, softmax, batch_loss, batch_metrics = sess.run([self.localizations, self.softmax, self.loss, self.metrics],
                                                                         feed_dict={self.inputs: batch_images, self.labels_enc: labels_enc})
            loss_mean = (loss_mean * step + batch_loss) / (step + 1)
            metrics_mean = (metrics_mean * step + batch_metrics) / (step + 1)
            step += 1

            # Convert output arrays to BoundingBox objects:
            batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, localizations, softmax)
            # Supress repeated predictions with non-maximum suppression:
            batch_pred_boxes = non_maximum_suppression_batched(batch_pred_boxes, self.opts.threshold_nms)
            # Supress repeated predictions with pc suppression:
            batch_pred_boxes = pc_suppression_batched(batch_pred_boxes, self.opts.threshold_pcs)
            # Mark True and False positives:
            batch_pred_boxes = mark_true_false_positives(batch_pred_boxes, batch_gt_boxes, self.opts.threshold_iou)
            all_gt_boxes.extend(batch_gt_boxes)
            all_pred_boxes.extend(batch_pred_boxes)

            if step % self.opts.nsteps_display == 0:
                print('Step %i / %i, loss: %.2e' % (step, reader.n_batches, batch_loss))

        precision, recall = compute_precision_recall_on_threshold(all_pred_boxes, all_gt_boxes, self.opts.th_conf)
        logging.info('Confidence threshold: ' + str(self.opts.th_conf) +
                     '. Precision: ' + str(precision) + '  -  recall: ' + str(recall))
        mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
        logging.info('Mean ' + split + ' Average Precision: ' + str(mAP))

        fintime = time.time()
        logging.debug('Done in %.2f s' % (fintime - initime))

        return loss_mean, metrics_mean

    # ------------------------------------------------------------------------------------------------------------------
    def train(self):

        print('')
        logging.info("Start training")
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
                current_lr = lr_scheduler.GetLearningRateAtEpoch(epoch, current_lr)
                _ = sess.run([self.update_lr_op], feed_dict={self.lr_to_update: current_lr})
                learning_rate = sess.run([self.learning_rate])[0]
                logging.info('Learning rate: ' + str(learning_rate))
                loss_mean = 0
                metrics_mean = np.zeros(shape=(self.arch.n_metrics))
                step = 0
                all_gt_boxes = []
                all_pred_boxes = []
                end_of_epoch = False
                iniepoch = time.time()
                while not end_of_epoch:
                    batch_images, batch_bboxes, batch_filenames, end_of_epoch = self.reader_train.get_next_batch(apply_data_aug=True)
                    labels_enc = self.arch.encode_gt_batched(batch_bboxes)

                    if self.nbatches_accum > 0:
                        if step % self.nbatches_accum == 0:
                            sess.run(self.zero_ops)
                        localizations, softmax, _, batch_loss, batch_metrics, summaryOut = \
                            sess.run([self.localizations, self.softmax, self.accum_ops, self.loss, self.metrics, merged],
                                     feed_dict={self.inputs: batch_images, self.labels_enc: labels_enc})
                        if (step + 1) % self.nbatches_accum == 0:
                            _ = sess.run([self.train_step])
                    else:
                        localizations, softmax, _, batch_loss, batch_metrics, summaryOut = \
                            sess.run([self.localizations, self.softmax, self.train_op, self.loss, self.metrics, merged],
                                     feed_dict={self.inputs: batch_images, self.labels_enc: labels_enc})

                    loss_mean = (loss_mean * step + batch_loss) / (step + 1)
                    metrics_mean = (metrics_mean * step + batch_metrics) / (step + 1)
                    step += 1
                    global_step += 1

                    if epoch % self.opts.nepochs_checktrain and not self.opts.recompute_train:
                        # Convert output arrays to BoundingBox objects:
                        batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, localizations, softmax)
                        # Supress repeated predictions with non-maximum suppression:
                        batch_pred_boxes = non_maximum_suppression_batched(batch_pred_boxes, self.opts.threshold_nms)
                        # Supress repeated predictions with pc suppression:
                        batch_pred_boxes = pc_suppression_batched(batch_pred_boxes, self.opts.threshold_pcs)
                        # Mark True and False positives:
                        batch_pred_boxes = mark_true_false_positives(batch_pred_boxes, batch_gt_boxes, self.opts.threshold_iou)
                        all_gt_boxes.extend(batch_gt_boxes)
                        all_pred_boxes.extend(batch_pred_boxes)

                    # Tensorboard:
                    summary_writer.add_summary(summaryOut, global_step)
                    if global_step == 1:
                        webbrowser.open_new_tab(tensorboard_url)

                    if step % self.opts.nsteps_display == 0:
                        print('Step %i / %i' % (step, self.reader_train.n_batches))

                finepoch = time.time()
                logging.debug('Epoch computed in %.2f s' % (finepoch - iniepoch))

                # Compute loss and metrics on training data:
                if epoch % self.opts.nepochs_checktrain == 0:
                    if self.opts.recompute_train:
                        loss_mean, metrics_mean = self.evaluate_on_dataset('train', sess)
                        logging.info('Train loss: %.2e' % loss_mean)
                        for m_idx in range(self.arch.n_metrics):
                            logging.info('Train ' + self.arch.metric_names[m_idx] + ': %.2f' % train_metrics[m_idx])
                    else:
                        precision, recall = compute_precision_recall_on_threshold(all_pred_boxes, all_gt_boxes, self.opts.th_conf)
                        logging.info('Confidence threshold: ' + str(self.opts.th_conf) + '. Precision: ' + str(precision) + '  -  recall: ' + str(recall))
                        mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
                        logging.info('Mean train Average Precision during epoch: ' + str(mAP))
                        logging.info('Mean train loss during epoch: %.2e' % loss_mean)
                        for m_idx in range(self.arch.n_metrics):
                            logging.info('Mean train ' + self.arch.metric_names[m_idx] + ' during epoch: %.2f' % metrics_mean[m_idx])
                    train_losses.append(loss_mean)
                    train_metrics.append(metrics_mean)

                # Compute loss and metrics on validation data:
                if epoch % self.opts.nepochs_checkval == 0:
                    val_loss, metrics = self.evaluate_on_dataset('val', sess)
                    val_losses.append(val_loss)
                    val_metrics.append(metrics)
                    logging.info('Val loss: %.2e' % val_loss)
                    for m_idx in range(self.arch.n_metrics):
                        logging.info('Val ' + self.arch.metric_names[m_idx] + ': %.2f' % metrics[m_idx])
                else:
                    val_loss = None

                # Plot training progress:
                if epoch % self.opts.nepochs_checktrain == 0 or epoch % self.opts.nepochs_checkval == 0:
                    tools.plot_training_history(train_metrics, train_losses, val_metrics, val_losses, self.arch.metric_names, self.opts, epoch)

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
        self.arch = MultiCellArch.MultiCellArch(self.opts.multi_cell_opts, self.nclasses, self.opts.outdir, self.opts.th_conf, self.classnames)
        self.define_inputs()
        self.arch.make_network(self.inputs)
        self.labels_enc = self.reader_train.build_labels_enc()
        self.loss, self.metrics, self.localizations, self.softmax = self.arch.make_loss_metrics_and_preds(self.labels_enc, self.filenames)
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
    def define_inputs(self):
        input_shape = self.arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader_train = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.arch, 'train')
            self.reader_val = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.arch, 'val')
            self.inputs = self.reader_train.build_inputs()

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

    def postprocess_gt_and_preds(self, bboxes, localizations, softmax):
        # bboxes: List of length batch_size, with elements (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        # localizations: (batch_size, nboxes, 4) [xmin, ymin, width, height]
        # softmax: (batch_size, nboxes, nclasses) [conf1, ..., confN]
        batch_gt_boxes = []
        batch_pred_boxes = []
        batch_size = len(bboxes)
        for img_idx in range(batch_size):
            # GT boxes:
            img_gt_boxes = []
            img_bboxes = bboxes[img_idx]
            for gt_idx in range(len(img_bboxes)):
                gt_box = BoundingBoxes.BoundingBox(img_bboxes[gt_idx, 1:5], img_bboxes[gt_idx, 0], img_bboxes[gt_idx, 5])
                img_gt_boxes.append(gt_box)
            batch_gt_boxes.append(img_gt_boxes)
            # Predicted boxes:
            img_pred_boxes = []
            for box_idx in range(self.arch.n_boxes):
                if self.opts.detect_against_background:
                    pred_class = np.argmax(softmax[img_idx, box_idx, :])
                    if pred_class != self.arch.background_id:
                        pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, softmax[img_idx, box_idx, pred_class], box_idx)
                        img_pred_boxes.append(pred_box)
                else:
                    pred_class = np.argmax(softmax[img_idx, box_idx, :-1])
                    conf = softmax[img_idx, box_idx, pred_class]
                    if conf > self.opts.th_conf_eval:
                        pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, conf, box_idx)
                        img_pred_boxes.append(pred_box)

            batch_pred_boxes.append(img_pred_boxes)

        return batch_gt_boxes, batch_pred_boxes

    def get_split_reader(self, split):
        if split == 'train':
            return self.reader_train
        elif split == 'val':
            return self.reader_val
        else:
            raise Exception('Unexpected split name: ' + str(split))

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


def pc_suppression_batched(batch_boxes, threshold_pcs):
    # batch_boxes: List of size batch_size, with lists with all the predicted bounding boxes in an image.
    batch_remaining_boxes = []
    batch_size = len(batch_boxes)
    for img_idx in range(batch_size):
        remaining_boxes = pc_suppression(batch_boxes[img_idx], threshold_pcs)
        batch_remaining_boxes.append(remaining_boxes)
    return batch_remaining_boxes


def pc_suppression(boxes, threshold_pcs):
    # boxes: List with all the predicted bounding boxes in the image.
    nboxes = len(boxes)
    boxes.sort(key=operator.attrgetter('area'))
    for i in range(nboxes):
        for j in range(i + 1, nboxes):
            if boxes[j].confidence >= boxes[i].confidence and np.abs(boxes[i].classid - boxes[j].classid) < 0.5:
                if tools.compute_pc(boxes[i].get_coords(), boxes[j].get_coords()) > threshold_pcs:
                    assert boxes[i].area <= boxes[j].area, 'Suppressing boxes in reverse order in PCS'
                    boxes[i].confidence = -np.inf
                    break
    remaining_boxes = [x for x in boxes if x.confidence != -np.inf]
    return remaining_boxes


def non_maximum_suppression_batched(batch_boxes, threshold_nms):
    # batch_boxes: List of size batch_size, with lists with all the predicted bounding boxes in an image.
    batch_remaining_boxes = []
    batch_size = len(batch_boxes)
    for img_idx in range(batch_size):
        remaining_boxes = non_maximum_suppression(batch_boxes[img_idx], threshold_nms)
        batch_remaining_boxes.append(remaining_boxes)
    return batch_remaining_boxes


def non_maximum_suppression(boxes, threshold_nms):
    # boxes: List with all the predicted bounding boxes in the image.
    nboxes = len(boxes)
    boxes.sort(key=operator.attrgetter('confidence'))
    for i in range(nboxes):
        for j in range(i + 1, nboxes):
            if np.abs(boxes[i].classid - boxes[j].classid) < 0.5:
                if tools.compute_iou(boxes[i].get_coords(), boxes[j].get_coords()) > threshold_nms:
                    assert boxes[i].confidence <= boxes[j].confidence, 'Suppressing boxes in reverse order in NMS'
                    boxes[i].confidence = -np.inf
                    break
    remaining_boxes = [x for x in boxes if x.confidence != -np.inf]
    return remaining_boxes


# This function must be called after marking the true and false positives.
def compute_precision_recall_on_threshold(pred_boxes, gt_boxes, th_conf):
    # pred_boxes (nimages) List with the predicted bounding boxes of each image.
    # gt_boxes (nimages) List with the ground truth boxes of each image.
    nimages = len(pred_boxes)
    TP = 0
    FP = 0
    FN = 0
    for img_idx in range(nimages):
        preds_image = pred_boxes[img_idx]
        gt_image = gt_boxes[img_idx]
        preds_over_th = [box for box in preds_image if box.confidence >= th_conf]
        TP_image = 0
        for box in preds_over_th:
            if box.tp == 'yes':
                TP_image += 1
            elif box.tp == 'no':
                FP += 1
            else:
                raise Exception('Prediction not determined as TP or FP.')
        TP += TP_image
        n_gt = len(gt_image)
        FN += n_gt - TP_image
    if TP + FP > 0:
        precision = float(TP) / (TP + FP)
    else:
        precision = 0
    if TP + FN > 0:
        recall = float(TP) / (TP + FN)
    else:
        recall = 0
    return precision, recall



def write_results(images, pred_boxes, gt_boxes, filenames, classnames, images_dir, batch_count):
    # images: List of length batch_size, with elements (input_width, input_height, 3)
    # pred_boxes: List of length batch_size, with lists of PredictedBox objects.
    # gt_boxes: List of length batch_size, with lists of BoundingBox objects.
    # filenames: List of length batch_size.
    batch_size = len(images)
    for img_idx in range(batch_size):
        img = images[img_idx]
        img = tools.add_mean_again(img)
        img = np.round(img).astype(np.uint8)
        this_preds = pred_boxes[img_idx]
        this_gt = gt_boxes[img_idx]
        name = filenames[img_idx]
        file_name = 'img' + str(img_idx) + '_' + name
        file_name = 'batch' + str(batch_count) + '_' + file_name
        dst_path = os.path.join(images_dir, file_name + '.png')
        draw_result(img, this_preds, this_gt, classnames)
        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        info_file_path = os.path.join(images_dir, file_name + '_info.txt')
        write_result_info(this_preds, img, info_file_path, classnames)


def write_result_info(pred_boxes, img, info_file_path, classnames):
    # pred_boxes: List of  PredictedBox objects.
    with open(info_file_path, 'w') as fid:
        height, width, _ = img.shape
        fid.write('[height, width] = ' + str([height, width]) + '\n')
        for box in pred_boxes:
            conf = box.confidence
            [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
            classid = int(box.classid)
            anc_idx = box.anc_idx
            fid.write('\n')
            fid.write('Box from anchor ' + str(anc_idx) + '\n')
            fid.write('[xmin, ymin, w, h]: ' + str([xmin, ymin, w, h]) + '\n')
            fid.write('classid = ' + str(classid) + ' (' + classnames[classid] + ')\n')
            fid.write('conf = ' + str(conf) + '\n')
            fid.write('tp = ' + str(box.tp) + '\n')


def draw_result(img, pred_boxes, gt_boxes, classnames):
    # pred_boxes: List of  PredictedBox objects.
    # gt_boxes: List of  BoundingBox objects.
    # Draw ground truth:
    for box in gt_boxes:
        [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
        cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
    # Draw predictions:
    for box in pred_boxes:
        conf = box.confidence
        [xmin, ymin, w, h] = box.get_abs_coords_cv(img)
        classid = int(box.classid)
        # Select color depending if the prediction is a true positive or not:
        if box.tp == 'yes':
            color = (0, 255, 0)
        elif box.tp == 'no':
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        # Draw box:
        cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), color, 2)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (125, 125, 125), -1)
        cv2.putText(img, classnames[classid] + ' : %.2f' % conf, (xmin + 5, ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return


def mark_true_false_positives(pred_boxes, gt_boxes, threshold_iou):
    nimages = len(pred_boxes)
    for i in range(nimages):
        gt_used = []
        pred_boxes[i].sort(key=operator.attrgetter('confidence'), reverse=True)
        for idx_pred in range(len(pred_boxes[i])):
            pred_boxes[i][idx_pred].set_tp(False)
            iou_vec = np.zeros(len(gt_boxes[i]))
            class_ids = np.zeros(len(gt_boxes[i]), dtype=np.int32)
            for idx_lab in range(len(gt_boxes[i])):
                iou_vec[idx_lab] = tools.compute_iou(pred_boxes[i][idx_pred].get_coords(), gt_boxes[i][idx_lab].get_coords())
                class_ids[idx_lab] = gt_boxes[i][idx_lab].classid
            ord_idx = np.argsort(-1 * iou_vec)
            iou_vec = iou_vec[ord_idx]
            class_ids = class_ids[ord_idx]
            for j in range(len(iou_vec)):
                if iou_vec[j] >= threshold_iou:
                    if pred_boxes[i][idx_pred].classid == class_ids[j]:
                        if ord_idx[j] not in gt_used:
                            pred_boxes[i][idx_pred].set_tp(True)
                            gt_used.append(ord_idx[j])
                            break
                else:
                    break
    return pred_boxes