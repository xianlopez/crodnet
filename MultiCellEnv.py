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
    def __init__(self, opts, split, HNM=False):

        self.opts = opts
        self.split = split
        self.HNM = HNM

        self.inputs = None
        self.restore_fn = None
        self.classnames = None
        self.nclasses = None
        self.reader = None

        if self.opts.detect_against_background:
            self.opts.th_conf = None

        # Initialize network:
        self.generate_graph()

    def evaluation_loop(self, save_path, detect_against_background, write_results):
        if write_results:
            images_dir = os.path.join(self.opts.outdir, 'images')
            os.makedirs(images_dir)
        with tf.Session(config=tools.get_config_proto(self.opts.gpu_memory_fraction)) as sess:
            self.saver.restore(sess, save_path)
            nbatches = self.reader.n_batches
            all_gt_boxes = []
            all_pred_boxes = []
            all_filenames = []
            step = 0
            end_of_epoch = False
            while not end_of_epoch:
                step += 1
                if step % self.opts.nsteps_display == 0:
                    print('Step %i / %i' % (step, nbatches))
                batch_images, batch_bboxes, batch_filenames, end_of_epoch = self.reader.get_next_batch()
                localizations, softmax = sess.run([self.localizations, self.softmax], feed_dict={self.inputs: batch_images})
                # Convert output arrays to BoundingBox objects:
                batch_gt_boxes, batch_pred_boxes = self.postprocess_gt_and_preds(batch_bboxes, localizations, softmax, detect_against_background)
                # Supress repeated predictions with non-maximum suppression:
                batch_pred_boxes = non_maximum_suppression_batched(batch_pred_boxes, self.opts.threshold_nms)
                # Supress repeated predictions with pc suppression:
                batch_pred_boxes = pc_suppression_batched(batch_pred_boxes, self.opts.threshold_pcs)
                # Mark True and False positives:
                batch_pred_boxes = mark_true_false_positives(batch_pred_boxes, batch_gt_boxes, self.opts.threshold_iou)
                if write_results:
                    boxes_to_show = self.apply_detection_filter(batch_pred_boxes)
                    write_results_fn(batch_images, boxes_to_show, batch_gt_boxes, batch_filenames, self.classnames, images_dir, step)
                all_gt_boxes.extend(batch_gt_boxes)
                all_pred_boxes.extend(batch_pred_boxes)
                all_filenames.extend(batch_filenames)
        return all_gt_boxes, all_pred_boxes, all_filenames

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_and_hnm(self, save_path):
        logging.info('Evaluating and looking for hard negatives.')
        detect_against_background = True
        initime = time.time()
        all_gt_boxes, all_pred_boxes, all_filenames = self.evaluation_loop(save_path, detect_against_background, self.opts.write_results)

        n_gt_boxes = count_gt_boxes(all_gt_boxes)
        max_hard_negatives = int(np.round(self.opts.hard_negatives_factor * float(n_gt_boxes)))
        hard_negatives = gather_hard_negatives(all_pred_boxes, max_hard_negatives)
        self.write_hard_negatives(hard_negatives, all_filenames)

        mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
        logging.info('Mean Average Precision: ' + str(mAP))
        fintime = time.time()
        logging.debug('Done in %.2f s' % (fintime - initime))

        return mAP

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate(self, save_path=None, compute_pr_on_th=False):
        initime = time.time()
        if save_path is None:
            save_path = self.opts.weights_file
        all_gt_boxes, all_pred_boxes, all_filenames = self.evaluation_loop(save_path, self.opts.detect_against_background, self.opts.write_results)

        mAP = mean_ap.compute_mAP(all_pred_boxes, all_gt_boxes, self.classnames, self.opts)
        logging.info('Mean Average Precision: ' + str(mAP))
        if compute_pr_on_th:
            precision, recall = compute_precision_recall_on_threshold(all_pred_boxes, all_gt_boxes, self.opts.th_conf)
            logging.info('Confidence threshold: ' + str(self.opts.th_conf) + '. Precision: ' + str(precision) + '  -  recall: ' + str(recall))
        fintime = time.time()
        logging.debug('Done in %.2f s' % (fintime - initime))

        return mAP

    def write_hard_negatives(self, hard_negatives, all_filenames):
        hn_dir = os.path.join(self.opts.outdir, 'hard_negatives')
        if not os.path.exists(hn_dir):
            os.makedirs(hn_dir)
        for hn in hard_negatives:
            filename = all_filenames[hn.img_idx]
            hn_ann_file = os.path.join(hn_dir, filename + '.txt')
            with open(hn_ann_file, 'a') as fid:
                fid.write(str(self.multi_cell_arch.background_id) +
                          ' {:5.3f} {:5.3f} {:5.3f} {:5.3f}\n'.format(hn.xmin, hn.ymin, hn.width, hn.height))
        if self.opts.debug_hnm:
            hn_imgs_dir = os.path.join(self.opts.outdir, 'hard_negatives_images')
            if not os.path.exists(hn_imgs_dir):
                os.makedirs(hn_imgs_dir)
            img_indices = [hn.img_idx for hn in hard_negatives]
            unique_img_indices = set(img_indices)
            for img_idx in unique_img_indices:
                filename = all_filenames[img_idx]
                img_path = os.path.join(self.opts.root_of_datasets, self.opts.dataset_name, 'images', filename + '.jpg')
                img = cv2.imread(img_path)
                hns_this_img = [hn for hn in hard_negatives if hn.img_idx == img_idx]
                draw_result(img, hns_this_img, [], self.classnames)
                dst_path = os.path.join(hn_imgs_dir, filename + '.jpg')
                cv2.imwrite(dst_path, img)

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
        self.localizations, self.softmax, self.common_representations, pc, dc, cm = self.multi_cell_arch.make(self.inputs)
        self.restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.opts.weights_file, tf.global_variables())
        self.saver = tf.train.Saver(name='net_saver', max_to_keep=1000000)

    def define_inputs_and_labels(self):
        input_shape = self.multi_cell_arch.get_input_shape()
        with tf.device('/cpu:0'):
            self.reader = MultiCellDataReader.MultiCellDataReader(input_shape, self.opts, self.multi_cell_arch, self.split)
            self.inputs = self.reader.build_inputs()

    def postprocess_gt_and_preds(self, bboxes, localizations, softmax, detect_against_background):
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
            for box_idx in range(self.multi_cell_arch.n_boxes):
                if detect_against_background:
                    pred_class = np.argmax(softmax[img_idx, box_idx, :])
                    if pred_class != self.multi_cell_arch.background_id:
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



def write_results_fn(images, pred_boxes, gt_boxes, filenames, classnames, images_dir, batch_count):
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


def count_gt_boxes(all_gt_boxes):
    n_gt = 0
    for img_idx in range(len(all_gt_boxes)):
        n_gt += len(all_gt_boxes[img_idx])
    return n_gt


def gather_hard_negatives(all_pred_boxes, max_num_to_keep):
    FPs = []
    for img_idx in range(len(all_pred_boxes)):
        for box in all_pred_boxes[img_idx]:
            if box.tp == 'no':
                box.set_img_idx(img_idx)
                FPs.append(box)
    FPs.sort(key=operator.attrgetter('confidence'), reverse=True)
    n_FPs = len(FPs)
    if n_FPs > 1:
        assert FPs[0].confidence >= FPs[1].confidence, 'FPs[0].confidence < FPs[1].confidence'
    if n_FPs > max_num_to_keep:
        FPs = FPs[:max_num_to_keep]
    return FPs