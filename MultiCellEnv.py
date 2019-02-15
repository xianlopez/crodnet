# ======================================================================================================================
import MultiCellDataReader
import tensorflow as tf
import tools
import logging
import time
import MultiCellArch
import os
import numpy as np
import BoundingBoxes
import mean_ap


# ======================================================================================================================
class TrainEnv:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, opts, split):

        self.opts = opts
        self.split = split

        self.inputs = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.reader = None

        # Initialize network:
        self.generate_graph()

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self):
        print('')
        logging.info('Start evaluation')
        with tf.Session(config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            self.restore_fn(sess)
            logging.info('Computing metrics on ' + self.split + ' data')
            initime = time.time()
            nbatches = self.reader.n_batches
            all_gt_boxes = []
            all_pred_boxes = []
            step = 0
            end_of_epoch = False
            while not end_of_epoch:
                step += 1
                if step % self.opts.nsteps_display == 0:
                    print('Step %i / %i' % (step, nbatches))
                batch_images, batch_bboxes, batch_filenames, end_of_epoch = self.reader.get_next_batch()
                predictions, CRs = sess.run([self.predictions, self.common_representations],
                                                              feed_dict={self.inputs: batch_images})
                # predictions = self.remove_repeated_predictions(predictions, CRs, sess)
                if self.opts.write_results:
                    write_results(batch_images, batch_bboxes, batch_filenames, predictions)
                batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, predictions)
                all_gt_boxes.extend(batch_gt_boxes)
                all_pred_boxes.extend(batch_pred_boxes)
            mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
            logging.info('Mean Average Precision: ' + str(mAP))
            fintime = time.time()
            logging.debug('Done in %.2f s' % (fintime - initime))

        return

    # ------------------------------------------------------------------------------------------------------------------
    def generate_graph(self):
        dirdata = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name)
        img_extension, self.classnames = tools.process_dataset_config(os.path.join(dirdata, 'dataset_info.xml'))
        self.nclasses = len(self.classnames)
        self.multi_cell_arch = MultiCellArch.MultiCellArch(self.opts.single_cell_opts, self.nclasses, self.opts.outdir)
        self.define_inputs_and_labels()
        self.predictions, self.common_representations = self.multi_cell_arch.make(self.inputs)
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, tf.global_variables())

    # ------------------------------------------------------------------------------------------------------------------
    def define_inputs_and_labels(self):
        input_shape = self.multi_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.multi_cell_arch, self.split)
            self.inputs = self.reader.build_inputs()


    def remove_repeated_predictions(self, predictions, CRs, sess):
        # predictions: (batch_size, nboxes, 4+nclasses) [xmin, ymin, width, height, conf1, ..., confN]
        # CRs: (batch_size, nboxes, lcr)

        initime = time.time()
        logging.debug('Removing repeated predictions...')

        all_confidences = predictions[..., 4:]  # (batch_size, nboxes, nclasses)
        pred_class = np.argmax(all_confidences, axis=-1)  # (batch_size, nboxes)
        max_conf = np.max(all_confidences, axis=-1)  # (batch_size, nboxes)
        mask_detections = np.logical_not(np.equal(pred_class, self.multi_cell_arch.background_id))

        batch_size = predictions.shape[0]
        boxes_indices = np.arange(self.multi_cell_arch.n_boxes)
        n_comparisons = 0
        for img_idx in range(batch_size):
            img_max_conf = max_conf[img_idx, :]  # (nboxes)
            img_mask = mask_detections[img_idx, :]  # (nboxes)
            conf_detections = np.extract(img_mask, img_max_conf)  # (ndetections)
            detections_indices = np.extract(img_mask, boxes_indices)  # (ndetections)
            sorting_indices = np.argsort(conf_detections)  # (ndetections)
            conf_detections = conf_detections[sorting_indices]  # (ndetections)
            if conf_detections[0] < conf_detections[1]:
                print('conf_detections[0] = ' + str(conf_detections[0]))
                print('conf_detections[1] = ' + str(conf_detections[1]))
                raise Exception('Sorted in reverse order.')
            detections_indices = detections_indices[sorting_indices]  # (ndetections)
            n_detections = conf_detections.shape[0]
            for i in range(n_detections-1):
                for j in range(i+1, n_detections):
                    idx1 = detections_indices[i]
                    idx2 = detections_indices[j]
                    class1 = pred_class[img_idx, idx1]
                    class2 = pred_class[img_idx, idx2]
                    if class1 == class2:
                        CR1 = CRs[img_idx, idx1, :]  # (lcr)
                        CR2 = CRs[img_idx, idx2, :]  # (lcr)
                        conf1 = conf_detections[i]
                        conf2 = conf_detections[j]
                        if conf1 < conf2:
                            print('idx1 = ' + str(idx1))
                            print('idx2 = ' + str(idx2))
                            print('conf1 = ' + str(conf1))
                            print('conf2 = ' + str(conf2))
                            raise Exception('conf1 < conf2 when removing repeated predictions')
                        comparison_2_values = sess.run([self.multi_cell_arch.comparison_op],
                                                       feed_dict={self.multi_cell_arch.CR1: CR1,
                                                                  self.multi_cell_arch.CR2: CR2})
                        comparison = comparison_2_values[0]
                        n_comparisons += 1
                        if comparison < self.opts.comp_th:
                            predictions[img_idx, idx2, 4:-1] = 0
                            predictions[img_idx, idx2, -1] = 1

        fintime = time.time()
        logging.info('Repeated predictions removed in %.2f s (%d comparisons).' % (fintime - initime, n_comparisons))
        return

    def postprocess_gt_and_preds(self, bboxes, predictions):
        # predictions: (batch_size, nboxes, 4+nclasses) [xmin, ymin, width, height, conf1, ..., confN]
        # bboxes: List of length batch_size, with elements (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        batch_gt_boxes = []
        batch_pred_boxes = []
        batch_size = len(bboxes)
        for img_idx in range(batch_size):
            # GT boxes:
            img_gt_boxes = []
            img_bboxes = bboxes[img_idx]
            for gt_idx in range(len(img_bboxes)):
                gt_box = BoundingBoxes.BoundingBox(img_bboxes[1:5], img_bboxes[0], img_bboxes[5])
                img_gt_boxes.append(gt_box)
            batch_gt_boxes.append(img_gt_boxes)
            # Predicted boxes:
            img_pred_boxes = []
            for box_idx in range(self.multi_cell_arch.n_boxes):
                pred_class = np.argmin(predictions[img_idx, box_idx, 4:])
                if pred_class != self.multi_cell_arch.background_id:
                    pred_box = BoundingBoxes.PredictedBox(predictions[img_idx, box_idx, :4], pred_class, predictions[img_idx, box_idx, 4+pred_class])
                    img_pred_boxes.append(pred_box)
            batch_pred_boxes.append(img_pred_boxes)
        return batch_gt_boxes, batch_pred_boxes


def write_results(images, bboxes, filenames, predictions):
    # images: List of length batch_size, with elements (input_width, input_height, 3)
    # bboxes: List of length batch_size, with elements (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
    # filenames: List of length batch_size
    # predictions: (batch_size, nboxes, 4+nclasses) [xmin, ymin, width, height, conf1, ..., confN]
    batch_size = len(images)
    for img_idx in range(batch_size):
        img = images[img_idx]
        gt = bboxes[img_idx]
        name = filenames[img_idx]
        # TODO