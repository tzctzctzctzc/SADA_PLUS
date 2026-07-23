# Modified from official EPIC-Kitchens action detection evaluation code
# see https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py
import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        valid = True
        for p_event in valid_events:
            if ((abs(s - p_event['segment'][0]) <= tol)
                    and (abs(e - p_event['segment'][1]) <= tol)
                    and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0, action_subsampling="all"):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    if split is None:
        splits = None
    elif isinstance(split, str):
        splits = {split.lower()}
    else:
        splits = {item.lower() for item in split}

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if splits is not None and v['subset'].lower() not in splits:
            continue
        # remove duplicated instances
        ants = remove_duplicate_annotations(v['annotations'])
        
        # for each event, grab the start/end time and label
        for event in ants:
            e_start = float(event['segment'][0])
            e_end = float(event['segment'][1])
            
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset ** i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            
            if action_subsampling == '10_majority_classes':
                if label_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    starts += [e_start]
                    stops += [e_end]
                    labels += [label_id]
                    vids += [k] # Add the video id
            elif action_subsampling == '20_majority_classes':
                if label_id in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,18,19]:
                    starts += [e_start]
                    stops += [e_end]
                    labels += [label_id]
                    vids += [k] # Add the video id
            elif action_subsampling == 'superclasses':
                assert False, "Not implemented yet"
            elif action_subsampling == 'all':
                starts += [e_start]
                stops += [e_end]
                labels += [label_id]
                vids += [k] # Add the video id
            else:
                assert False, "Unknown action subsampling method"
            
    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id': vids,
        't-start': starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset ** i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id': vids,
        't-start': starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
            self,
            ant_file,
            split=None,
            tiou_thresholds=np.linspace(0.1, 0.5, 5),
            label='label_id',
            label_offset=0,
            num_workers=8,
            dataset_name=None,
            action_subsampling="all"
    ):

        self.tiou_thresholds = tiou_thresholds
        self.ap = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        self.ground_truth = load_gt_seg_from_json( 
            ant_file, split=self.split, label=label, label_offset=label_offset, action_subsampling=action_subsampling)
        
        # remove labels that do not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.idx_to_activity_index = {i: j for i, j in enumerate(sorted(self.ground_truth['label'].unique()))} # Used to go back to the original action label space 
        self.ground_truth['label'] = self.ground_truth['label'].replace(self.activity_index)

        # remove action instances with length=0
        self.ground_truth = self.ground_truth.drop(
            np.where(self.ground_truth['t-start'] == self.ground_truth['t-end'])[0])
        
        
    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))    # len(activity_index) == n_classes to consider
        tp = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        fp = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        fn = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        #f1 = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            # Returns ap, tp, fp, fn
            ap[:, cidx] = results[i][0]
            tp[:, cidx] = results[i][1]
            fp[:, cidx] = results[i][2]
            fn[:, cidx] = results[i][3]
            
        return ap, tp, fp, fn

    def evaluate(self, preds, verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id': preds['video-id'],
                't-start': preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        # always reset ap

        # make the label ids consistent
        preds['label'] = preds['label'].replace(self.activity_index)

        # compute mAP
        self.ap, self.tp, self.fp, self.fn = self.wrapper_compute_average_precision(preds)
        
        # Compute the f1 score
        n_tp = self.tp.sum(axis=1)  # shape (n_tiou_thresholds,)
        n_fp = self.fp.sum(axis=1) # shape (n_tiou_thresholds,)
        n_fn = self.fn.sum(axis=1) # shape (n_tiou_thresholds,)
        
        rec = n_tp / (n_tp + n_fn + 1e-16)
        prec = n_tp / (n_tp + n_fp)
        f1_score = 2.0 * (prec * rec) / (prec + rec)
        f1 = np.nan_to_num(f1_score) * 100  # shape (n_tiou_thresholds,)
        
        
        # Fix error: If any of the values is nan, then set to 0
        self.ap = np.nan_to_num(self.ap)
        
        # Average over the different intervals, to obtain class-wise mAP
        class_mAP = self.ap.mean(axis=0)    # shape: (n_classes,)
        
        # Average over the different classes
        mAP = self.ap.mean(axis=1)
        
        # Compute the f1 score
        n_tp = self.tp.sum(axis=1)  # shape (n_tiou_thresholds,)
        n_fp = self.fp.sum(axis=1) # shape (n_tiou_thresholds,)
        n_fn = self.fn.sum(axis=1) # shape (n_tiou_thresholds,)
        
        # Compute the precision and recall at different tiou thresholds
        rec = n_tp / (n_tp + n_fn + 1e-16)
        prec = n_tp / (n_tp + n_fp)
        
        f1_score = 2.0 * (prec * rec) / (prec + rec)
        f1 = np.nan_to_num(f1_score) * 100  # shape (n_tiou_thresholds,)
        
    
        # Final metrics
        average_mAP = mAP.mean()
        average_f1 = f1.mean()
        average_recall = rec.mean()
        average_precision = prec.mean()


        # Compute the weighted AP, F1, precision and recall (weighted average according to the number of true instances)
        class_counts = self.tp + self.fn  # shape: (n_tiou_thresholds, n_classes)
        weighted_ap = np.sum(self.ap * class_counts) / np.sum(class_counts)

        # print results
        if verbose:
            # print the results
            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            block = ''
            for tiou, tiou_mAP, tiou_mf1 in zip(self.tiou_thresholds, mAP, f1):
                block += '\n|tIoU = {:.2f}: mAP = {:.2f} (%):  mf1 = {:.2f}'.format(tiou, tiou_mAP * 100, tiou_mf1)
            print(block)
            print('Avearge mAP: {:.2f} (%)'.format(average_mAP * 100))
            print('Avearge mf1: {:.2f} (%)'.format(average_f1))

        # return the results
        return mAP, average_mAP, class_mAP, f1, average_f1, rec, average_recall, prec, average_precision, weighted_ap


def compute_average_precision_detection(
        ground_truth,
        prediction,
        tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    tp_total = np.zeros(len(tiou_thresholds))
    fp_total = np.zeros(len(tiou_thresholds))
    fn_total = np.zeros(len(tiou_thresholds))
    
    if prediction.empty:
        return ap

    npos = float(len(ground_truth)) # TODO: Check that this is consistent!
    
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    
    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            # Iterate over each of the predictions
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1
                
    # Compute the fn (i.e., the number of gt that are left unassigned)
    fn_total = np.full((len(tiou_thresholds),), len(ground_truth)) - np.sum(tp, axis=1)
            
            
    # Compute the mAP score
    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

        # Compute the F1 score
        tp_total[tidx] = np.sum(tp[tidx, :])
        fp_total[tidx] = np.sum(fp[tidx, :])

    return ap, tp_total, fp_total, fn_total


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union

    if np.any(segments_union == 0) or np.any(np.isnan(segments_union)):
        print(1)
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
