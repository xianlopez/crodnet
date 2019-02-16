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
import cv2
import operator


# ======================================================================================================================
class MultiCellEnv:

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
        if self.opts.write_results:
            images_dir = os.path.join(self.opts.outdir, 'images')
            os.makedirs(images_dir)
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
                localizations, sofmtax, CRs = sess.run([self.localizations, self.softmax, self.common_representations],
                                                       feed_dict={self.inputs: batch_images})
                # sofmtax = self.remove_repeated_predictions(sofmtax, CRs, sess)
                batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, localizations, sofmtax)
                batch_pred_boxes = non_maximum_suppression_batched(batch_pred_boxes, self.opts.threshold_nms)
                if self.opts.write_results:
                    write_results(batch_images, batch_pred_boxes, batch_gt_boxes, batch_filenames, self.classnames, images_dir)
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
        self.multi_cell_arch = MultiCellArch.MultiCellArch(self.opts.multi_cell_opts, self.nclasses)
        self.define_inputs_and_labels()
        self.localizations, self.softmax, self.common_representations = self.multi_cell_arch.make(self.inputs)
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, tf.global_variables())

    def define_inputs_and_labels(self):
        input_shape = self.multi_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.multi_cell_arch, self.split)
            self.inputs = self.reader.build_inputs()


    def remove_repeated_predictions(self, sofmtax, CRs, sess):
        # TODO: Non me acaba de convencer como esta feita esta funcion...
        # sofmtax: (batch_size, nboxes, nclasses) [conf1, ..., confN]
        # CRs: (batch_size, nboxes, lcr)

        initime = time.time()
        logging.debug('Removing repeated predictions...')

        pred_class = np.argmax(sofmtax, axis=-1)  # (batch_size, nboxes)
        max_conf = np.max(sofmtax, axis=-1)  # (batch_size, nboxes)
        mask_detections = np.logical_not(np.equal(pred_class, self.multi_cell_arch.background_id))

        batch_size = sofmtax.shape[0]
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
                            sofmtax[img_idx, idx2, :-1] = 0
                            sofmtax[img_idx, idx2, -1] = 1

        fintime = time.time()
        logging.info('Repeated predictions removed in %.2f s (%d comparisons).' % (fintime - initime, n_comparisons))
        return sofmtax

    def postprocess_gt_and_preds(self, bboxes, localizations, sofmtax):
        # bboxes: List of length batch_size, with elements (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        # localizations: (batch_size, nboxes, 4) [xmin, ymin, width, height]
        # sofmtax: (batch_size, nboxes, nclasses) [conf1, ..., confN]
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
            for box_idx in range(self.multi_cell_arch.n_boxes):
                pred_class = np.argmax(sofmtax[img_idx, box_idx, :])
                if pred_class != self.multi_cell_arch.background_id:
                    pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, sofmtax[img_idx, box_idx, pred_class])
                    img_pred_boxes.append(pred_box)
            batch_pred_boxes.append(img_pred_boxes)

        # Mark True and False positives:
        batch_pred_boxes = mark_true_false_positives(batch_pred_boxes, batch_gt_boxes, self.opts.threshold_iou)

        return batch_gt_boxes, batch_pred_boxes


def non_maximum_suppression_batched(batch_boxes, threshold_nms):
    batch_remaining_boxes = []
    batch_size = len(batch_boxes)
    for img_idx in range(batch_size):
        remaining_boxes = non_maximum_suppression(batch_boxes[img_idx], threshold_nms)
        batch_remaining_boxes.append(remaining_boxes)
    return batch_remaining_boxes


def non_maximum_suppression(boxes, threshold_nms):
    # boxes: List with all the predicted bounding boxes in the image.
    nboxes = len(boxes)
    boxes.sort(key=operator.attrgetter('confidence'), reverse=True)
    for i in range(nboxes):
        if boxes[i].confidence != -np.inf:
            for j in range(i + 1, nboxes):
                if boxes[j].confidence != -np.inf:
                    if tools.compute_iou(boxes[i].get_coords(), boxes[j].get_coords()) > threshold_nms:
                        boxes[j].confidence = -np.inf
    remaining_boxes = [x for x in boxes if x.confidence != -np.inf]
    return remaining_boxes


def write_results(images, pred_boxes, gt_boxes, filenames, classnames, images_dir):
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
        dst_path = os.path.join(images_dir, name + '.png')
        draw_result(img, this_preds, this_gt, classnames)
        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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
        cv2.rectangle(img, (xmin, ymin - 20),
                      (xmin + w, ymin), (125, 125, 125), -1)
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