import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import tools
import operator
import os

class MeanAPOpts:
    npoints_interp = 100
    epsilon = 1e-6
    th_samples = [0.3, 0.5, 0.7]


class PredictionMatch:
    def __init__(self, confidence, matches_gt_box):
        self.confidence = confidence
        self.matches_gt_box = matches_gt_box


def plot_pr_curve(recall, precision_rect, recall_interp, precision_interp, thresholds, classname, AP, outdir, mean_ap_opts):
    plt.figure()
    plt.plot(recall, precision_rect, 'b-')
    plt.plot(recall_interp, precision_interp, 'r:')
    for th in mean_ap_opts.th_samples:
        idx = int(np.argmin(np.abs(np.array(thresholds) - th)))
        th_near = thresholds[idx]
        precision_at_th = precision_rect[idx]
        recall_at_th = recall[idx]
        plt.plot(recall_at_th, precision_at_th, 'k*')
        plt.text(recall_at_th, precision_at_th, str(th_near))
    plt.xlim((0, 1))
    plt.ylim((-0.1, 1.1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(classname + ' - AP: ' + str(AP))
    fig_path = os.path.join(outdir, 'PR_' + classname + '.png')
    plt.savefig(fig_path)


def precision_recall_curve(predictions_matches, n_objects):
    # logging.debug('Computing precision-recall curve...')
    ini = time.time()
    predictions_matches.sort(key=operator.attrgetter('confidence'), reverse=True)
    n_points = len(predictions_matches)
    precision_vec = np.zeros(shape=(n_points), dtype=np.float32)
    recall_vec = np.zeros(shape=(n_points), dtype=np.float32)
    threshold_vec = np.zeros(shape=(n_points), dtype=np.float32)
    FP = 0
    TP = 0
    n_predictions = 0
    # count = 0
    for i in range(n_points):
        prediction = predictions_matches[i]
        # count += 1
        # print(str(count) + '/' + str(n_points))
        n_predictions += 1
        if prediction.matches_gt_box:
            TP += 1
        else:
            FP += 1
        # FN = n_objects - TP
        precision = TP / n_predictions
        if n_objects == 0:
            recall = 0
        else:
            recall = TP / n_objects
        precision_vec[i] = precision
        recall_vec[i] = recall
        threshold_vec[i] = prediction.confidence
    lapse = time.time() - ini
    # logging.debug('Precision-recall curve done in ' + str(lapse) + ' s.')
    return threshold_vec, recall_vec, precision_vec


# Rectified precision.
# At a given point, this is the maximum precision at that point and all with a higher recall.
def rectify_precision(precision):
    # logging.debug('Rectifying precision...')
    ini = time.time()
    precision_rect = np.zeros_like(precision)
    npoints = len(precision)
    max_precision = 0
    for i in range(npoints - 1, -1, -1):
        this_precision = precision[i]
        if this_precision > max_precision:
            max_precision = this_precision
        precision_rect[i] = max_precision
    lapse = time.time() - ini
    # logging.debug('Precision rectified in ' + str(lapse) + ' s.')
    return precision_rect


# Interpolate precision-recall curve:
def interpolate_pr_curve(precision, recall, mean_ap_opts):
    # logging.debug('Interpolating PR curve...')
    ini = time.time()
    if np.max(recall) < 1 - mean_ap_opts.epsilon:
        # recall.append(np.max(recall) + mean_ap_opts.epsilon)
        # precision.append(0)
        recall = np.concatenate([recall, np.array([np.max(recall) + mean_ap_opts.epsilon])], axis=0)
        precision = np.concatenate([precision, np.array([0])], axis=0)
    recall_interp = np.linspace(0, 1, mean_ap_opts.npoints_interp)
    precision_interp = np.interp(recall_interp, recall, precision)
    lapse = time.time() - ini
    # logging.debug('PR curve interpolated in ' + str(lapse) + ' s.')
    return recall_interp, precision_interp


def compute_mAP(predictions, labels, classnames, opts):
    # predictions (nimages) List with the predicted bounding boxes of each image.
    # labels (nimages) List with the ground truth boxes of each image.
    logging.debug('Computing mean average precision...')
    initime = time.time()

    nclasses = len(classnames)
    nimages = len(predictions)

    predictions_matches = []
    for cl in range(nclasses):
        predictions_matches.append([])

    nobj_allclasses = np.zeros(shape=(nclasses), dtype=np.int32)

    # Compute correspondences between predictions and ground truth for every image.
    # logging.debug('Computing correspondences...')
    ini = time.time()
    for i in range(nimages):
        # print(str(i) + '/' + str(nimages))
        predboxes_img = predictions[i]
        gtlist_img = labels[i]
        for cl in range(nclasses):
            gtlist_img_class = [box for box in gtlist_img if box.classid == cl]
            predboxes_img_class = [box for box in predboxes_img if box.classid == cl]
            nobj_allclasses[cl] += len(gtlist_img_class)
            gt_used = []
            predboxes_img_class.sort(key=operator.attrgetter('confidence'), reverse=True)
            for k in range(len(predboxes_img_class)):
                matches_gt = False
                iou_list = []
                iou_idx = []
                # Compute iou with all gt boxes
                for l in range(len(gtlist_img_class)):
                    if l not in gt_used:
                        iou = tools.compute_iou(predboxes_img_class[k].get_coords(), gtlist_img_class[l].get_coords())
                        if iou >= opts.threshold_iou_map:
                            iou_list.append(iou)
                            iou_idx.append(l)
                if len(iou_list) > 0:
                    iou_array = np.array(iou_list)
                    # Order iou in descending order:
                    ord_idx = np.argsort(-1 * iou_array)
                    iou_idx = np.array(iou_idx)[ord_idx]
                    # Assign ground truth box:
                    for l in range(len(iou_idx)):
                        if iou_idx[l] not in gt_used:
                            gt_used.append(iou_idx[l])
                            matches_gt = True
                            break
                if matches_gt:
                    predictions_matches[cl].append(PredictionMatch(predboxes_img_class[k].confidence, True))
                else:
                    predictions_matches[cl].append(PredictionMatch(predboxes_img_class[k].confidence, False))
    lapse = time.time() - ini
    # logging.debug('Correspondences ready (done in ' + str(lapse) + ' s).')

    # Compute precision and recall curves for every class:
    precision_allclasses = []
    precision_rec_allclasses = []
    recall_allclasses = []
    AP_allclasses = []
    for cl in range(nclasses):
        # Compute the precision-recall curve:
        thresholds, recall, precision = precision_recall_curve(predictions_matches[cl], nobj_allclasses[cl])

        if len(recall) > 0:

            # Rectify precision:
            precision_rect = rectify_precision(precision)

            # Interpolate precision-recall curve:
            recall_interp, precision_interp = interpolate_pr_curve(precision_rect, recall, opts.mean_ap_opts)

            # Average precision:
            AP = 1 / len(recall_interp) * np.sum(precision_interp)

            # Plot curve:
            plot_pr_curve(recall, precision_rect, recall_interp, precision_interp, thresholds, classnames[cl], AP, opts.outdir, opts.mean_ap_opts)

            precision_allclasses.append(precision)
            precision_rec_allclasses.append(precision_rect)
            recall_allclasses.append(recall)

        else:
            AP = 0

        AP_allclasses.append(AP)
        # logging.info('class ' + classnames[cl] + '  - ' + 'AP: ' + str(AP))

    # Mean average precision:
    mAP = 0
    for i in range(nclasses):
        mAP += AP_allclasses[i]
    mAP = mAP / nclasses
    logging.info('Mean average precision: ' + str(mAP))

    fintime = time.time()
    logging.debug('mAP computed in %.2f s' % (fintime - initime))

    return mAP
