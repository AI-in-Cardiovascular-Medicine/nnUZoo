"""
Created on Thu Feb 17 18:10:52 2025
adapted form https://github.com/bowang-lab/U-Mamba/blob/main/evaluation/compute_cell_metric.py and modified to create
test_summary.json like the one nnunetv2 generates
"""

import argparse
import os

import numpy as np
import tifffile as tif
from deep_utils import JsonUtils
from numba import jit
from scipy.optimize import linear_sum_assignment
from skimage import segmentation, io, measure

join = os.path.join
from tqdm import tqdm


def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()

    # preallocate a 'contact map' matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def dice(gt, seg):
    if np.count_nonzero(gt) == 0 and np.count_nonzero(seg) == 0:
        dice_score = 1.0
    elif np.count_nonzero(gt) == 0 and np.count_nonzero(seg) > 0:
        dice_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(gt) + np.count_nonzero(seg)
        dice_score = 2 * union / intersection
    return dice_score


def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0

    return tp, fp, fn


def remove_boundary_cells(mask):
    "We do not consider boundary cells during evaluation"
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2:W - 2, 2:H - 2] = 0
    bd_cells = np.unique(mask * bd)
    for i in bd_cells[1:]:
        mask[mask == i] = 0
    new_label, _, _ = segmentation.relabel_sequential(mask)
    return new_label


# def main():
parser = argparse.ArgumentParser('Compute F1 score for cell segmentation results', add_help=False)
# Dataset parameters
parser.add_argument('-g', '--gt_path',
                    default='./NNUNET_OUTPUT/nnunet_raw/Dataset032_NeurlPSCell/labelsVar',
                    type=str, help='path to ground truth')
parser.add_argument('-s', '--seg_path', type=str,
                    default='./NNUNET_OUTPUT/nnunet_raw/Dataset032_NeurlPSCell/imagesTs_nnUNetTrainer_Pred',
                    help='path to segmentation results; file names are the same as ground truth', required=False)
parser.add_argument('-n', '--save_path',
                    default='./NNUNET_OUTPUT/nnunet_raw/Dataset032_NeurlPSCell/imagesTs_nnUNetTrainer_Pred/test_summary.json',
                    type=str, help='name of the json file to save the results')
parser.add_argument('--count_bd_cells', default=False, action='store_true', required=False,
                    help='remove the boundary cells when computing metrics by default')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path

names = sorted(os.listdir(seg_path))
names = [i for i in names if i.endswith('.png')]
names = [i for i in names if os.path.isfile(join(gt_path, i.split('.png')[0] + '_label.tiff'))]
print('num of files:', len(names))

if __name__ == '__main__':
    threshold = 0.5
    print('compute metrics at threshold:', threshold)
    metrics = dict(foreground_mean=dict(Dice=0),
                   metric_per_case=[])
    mean_dice = []
    for name in tqdm(names):
        reference_file = join(gt_path, name.split('.png')[0] + '_label.tiff')
        prediction_file = join(seg_path, name)
        gt = tif.imread(reference_file)
        seg = io.imread(prediction_file)
        seg = measure.label(seg == 1)
        gt[gt > 0] = 1
        seg[seg > 0] = 1
        dice_score = dice(gt, seg)
        mean_dice.append(dice_score)
        metrics['metric_per_case'].append(dict(metrics={"1": dice_score},
                                               prediction_file=prediction_file,
                                               reference_file=reference_file))

    metrics['foreground_mean']["Dice"] = float(np.mean(mean_dice))
    JsonUtils.dump(args.save_path, metrics)
