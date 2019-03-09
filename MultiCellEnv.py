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

        if self.opts.detect_against_background:
            self.opts.th_conf = None

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
                if self.opts.multi_cell_opts.predict_cm:
                    localizations, softmax, CRs, cm = sess.run([self.localizations, self.softmax, self.common_representations, self.cm],
                                                           feed_dict={self.inputs: batch_images})
                else:
                    localizations, softmax, CRs = sess.run([self.localizations, self.softmax, self.common_representations],
                                                           feed_dict={self.inputs: batch_images})
                    cm = None
                # softmax = self.remove_repeated_predictions(softmax, CRs, sess)
                # Convert output arrays to BoundingBox objects:
                batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, localizations, softmax, cm)
                # Supress repeated predictions with non-maximum suppression:
                batch_pred_boxes = non_maximum_suppression_batched(batch_pred_boxes, self.opts.threshold_nms)
                # Supress repeated predictions with pc suppression:
                batch_pred_boxes = pc_suppression_batched(batch_pred_boxes, self.opts.threshold_pcs)
                # Supress repeated predictions that non-maximum supression using Centrality Measure:
                # batch_pred_boxes = self.nms_cm_batched(batch_pred_boxes)
                # Supress repeated predictions with comparisons:
                # batch_pred_boxes = self.remove_repeated_predictions_batched(batch_pred_boxes, CRs, sess)
                # Mark True and False positives:
                batch_pred_boxes = mark_true_false_positives(batch_pred_boxes, batch_gt_boxes, self.opts.threshold_iou)
                if self.opts.write_results:
                    boxes_to_show = self.apply_detection_filter(batch_pred_boxes)
                    write_results(batch_images, boxes_to_show, batch_gt_boxes, batch_filenames, self.classnames, images_dir, step)
                all_gt_boxes.extend(batch_gt_boxes)
                all_pred_boxes.extend(batch_pred_boxes)
            precision, recall = compute_precision_recall_on_threshold(all_pred_boxes, all_gt_boxes, self.opts.th_conf)
            logging.info('Confidence threshold: ' + str(self.opts.th_conf) + '. Precision: ' + str(precision) + '  -  recall: ' + str(recall))
            mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
            logging.info('Mean Average Precision: ' + str(mAP))
            fintime = time.time()
            logging.debug('Done in %.2f s' % (fintime - initime))

        return

    def apply_detection_filter(self, batch_pred_boxes):
        boxes_to_show = []
        for img_idx in range(len(batch_pred_boxes)):
            boxes_to_show_this_image = []
            boxes_this_image = batch_pred_boxes[img_idx]
            for box in boxes_this_image:
                if box.confidence >= self.opts.th_conf:
                    boxes_to_show_this_image.append(box)
            boxes_to_show.append(boxes_to_show_this_image)
        return boxes_to_show

    # ------------------------------------------------------------------------------------------------------------------
    def generate_graph(self):
        dirdata = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name)
        img_extension, self.classnames = tools.process_dataset_config(os.path.join(dirdata, 'dataset_info.xml'))
        self.nclasses = len(self.classnames)
        self.multi_cell_arch = MultiCellArch.MultiCellArch(self.opts.multi_cell_opts, self.nclasses, self.opts.outdir, self.opts.th_conf, self.classnames)
        self.define_inputs_and_labels()
        self.localizations, self.softmax, self.common_representations, pc, dc, self.cm = self.multi_cell_arch.make(self.inputs)
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, tf.global_variables())

    def define_inputs_and_labels(self):
        input_shape = self.multi_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.multi_cell_arch, self.split)
            self.inputs = self.reader.build_inputs()

    def remove_repeated_predictions_batched(self, batch_boxes, batch_CRs, sess):
        # batch_boxes: List of size batch_size, with lists with all the predicted bounding boxes in an image.
        # batch_CRs: (batch_size, nboxes, lcr)
        initime = time.time()
        # logging.debug('Removing repeated predictions...')
        batch_remaining_boxes = []
        batch_size = len(batch_boxes)
        n_comparisons = 0
        n_detections_initially = 0
        n_detections_finally = 0
        for img_idx in range(batch_size):
            n_detections_initially += len(batch_boxes[img_idx])
            remaining_boxes, n_comparisons_img = self.remove_repeated_predictions(batch_boxes[img_idx], batch_CRs[img_idx, ...], sess)
            n_detections_finally += len(remaining_boxes)
            batch_remaining_boxes.append(remaining_boxes)
        fintime = time.time()
        logging.info('Repeated predictions removed in %.2f s (%d comparisons).' % (fintime - initime, n_comparisons))
        # print('Number of images: ' + str(batch_size))
        # print('Initial number of detections: ' + str(n_detections_initially))
        # print('Initial number of detections: ' + str(n_detections_finally))
        # print('In relative terms, we passed from ' + str(n_detections_initially / float(batch_size)) +
        #       ' to ' + str(n_detections_finally / float(batch_size)) + ' detections per image.')
        return batch_remaining_boxes

    def remove_repeated_predictions(self, boxes, CRs, sess):
        # boxes: List with all the predicted bounding boxes in the image.
        # CRs: (nboxes, lcr)

        nboxes = len(boxes)
        boxes.sort(key=operator.attrgetter('confidence'))

        n_comparisons = 0
        for i in range(nboxes):
            box1 = boxes[i]
            class1 = box1.classid
            conf1 = box1.confidence
            anc_idx_1 = box1.anc_idx
            CR1 = CRs[anc_idx_1, :]  # (lcr)
            list_to_compare = self.multi_cell_arch.comparisons_references[anc_idx_1]
            for j in range(i+1, nboxes):
                box2 = boxes[j]
                anc_idx_2 = box2.anc_idx
                if anc_idx_2 in list_to_compare:
                    class2 = box2.classid
                    if class1 == class2:
                        CR2 = CRs[anc_idx_2, :]  # (lcr)
                        conf2 = box2.confidence
                        if conf1 > conf2:
                            print('idx1 = ' + str(anc_idx_1))
                            print('idx2 = ' + str(anc_idx_2))
                            print('conf1 = ' + str(conf1))
                            print('conf2 = ' + str(conf2))
                            raise Exception('conf1 > conf2 when removing repeated predictions')
                        comparison_2_values = sess.run([self.multi_cell_arch.comparison_op],
                                                       feed_dict={self.multi_cell_arch.CR1: CR1,
                                                                  self.multi_cell_arch.CR2: CR2})
                        comparison = comparison_2_values[0][0]
                        n_comparisons += 1
                        if comparison < self.opts.comp_th:
                            box1.confidence = -np.inf
                            break

        remaining_boxes = [x for x in boxes if x.confidence != -np.inf]

        return remaining_boxes, n_comparisons

    def postprocess_gt_and_preds(self, bboxes, localizations, softmax, cm):
        # bboxes: List of length batch_size, with elements (n_gt, 7) [class_id, x_min, y_min, width, height, pc, gt_idx]
        # localizations: (batch_size, nboxes, 4) [xmin, ymin, width, height]
        # softmax: (batch_size, nboxes, nclasses) [conf1, ..., confN]
        # cm: (batch_size, nboxes) or None
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
                if self.opts.detect_against_background:
                    pred_class = np.argmax(softmax[img_idx, box_idx, :])
                    if pred_class != self.multi_cell_arch.background_id:
                        if self.opts.multi_cell_opts.predict_cm:
                            if cm[img_idx, box_idx] > self.opts.th_cm_low:
                                pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class,
                                                                      softmax[img_idx, box_idx, pred_class], box_idx, cm=cm[img_idx, box_idx])
                                img_pred_boxes.append(pred_box)
                        else:
                            pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, softmax[img_idx, box_idx, pred_class], box_idx)
                            img_pred_boxes.append(pred_box)
                else:
                    pred_class = np.argmax(softmax[img_idx, box_idx, :-1])
                    conf = softmax[img_idx, box_idx, pred_class]
                    if self.opts.multi_cell_opts.predict_cm:
                        if conf > self.opts.th_conf_eval and cm[img_idx, box_idx] > self.opts.th_cm_low:
                            pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, conf, box_idx, cm=cm[img_idx, box_idx])
                            img_pred_boxes.append(pred_box)
                    else:
                        if conf > self.opts.th_conf_eval:
                            pred_box = BoundingBoxes.PredictedBox(localizations[img_idx, box_idx], pred_class, conf, box_idx)
                            img_pred_boxes.append(pred_box)

            batch_pred_boxes.append(img_pred_boxes)

        return batch_gt_boxes, batch_pred_boxes

    def nms_cm_batched(self, batch_boxes):
        # batch_boxes: List of size batch_size, with lists with all the predicted bounding boxes in an image.
        batch_remaining_boxes = []
        batch_size = len(batch_boxes)
        for img_idx in range(batch_size):
            remaining_boxes = self.nms_cm(batch_boxes[img_idx])
            batch_remaining_boxes.append(remaining_boxes)
        return batch_remaining_boxes

    def nms_cm(self, boxes):
        # boxes: List with all the predicted bounding boxes in the image.
        boxes_that_can_be_suppressed = []
        boxes_that_cannot_be_suppressed = []
        for box in boxes:
            if box.cm < self.opts.th_cm_high:
                boxes_that_can_be_suppressed.append(box)
            else:
                boxes_that_cannot_be_suppressed.append(box)
        boxes_that_can_be_suppressed.sort(key=operator.attrgetter('cm'), reverse=True)
        nboxes = len(boxes_that_can_be_suppressed)
        for i in range(nboxes):
            if boxes_that_can_be_suppressed[i].confidence != -np.inf:
                for j in range(i + 1, nboxes):
                    if boxes_that_can_be_suppressed[j].confidence != -np.inf:
                        if tools.compute_iou(boxes_that_can_be_suppressed[i].get_coords(), boxes_that_can_be_suppressed[j].get_coords()) > self.opts.cm_iou:
                            assert boxes_that_can_be_suppressed[i].cm >= boxes_that_can_be_suppressed[j].cm, 'Suppressing boxes in reverse order in NMS-CM'
                            boxes_that_can_be_suppressed[j].confidence = -np.inf
        remaining_boxes = [x for x in boxes_that_can_be_suppressed if x.confidence != -np.inf]
        remaining_boxes = remaining_boxes + boxes_that_cannot_be_suppressed
        remaining_boxes.sort(key=operator.attrgetter('confidence'), reverse=True)
        return remaining_boxes


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