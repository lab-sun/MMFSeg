# By Zhen Feng, Sep. 30, 2025
# Email: zfeng94@outlook.com

import numpy as np 
from PIL import Image 
import torch
 
# 0:unlabeled, 1:pothole
def get_palette():
    unlabelled = [0,0,0]
    pothole        = [153,0,0]
    palette    = np.array([unlabelled,pothole])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(weight_name+'/Pred_' + weight_name + '_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    F1_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN
        if (recall_per_class[cid] == np.nan) | (precision_per_class[cid] == np.nan) |(precision_per_class[cid]==0)|(recall_per_class[cid]==0):
            F1_per_class[cid] = np.nan
        else :
            F1_per_class[cid] = 2 / (1/precision_per_class[cid] +1/recall_per_class[cid])

    return precision_per_class, recall_per_class, iou_per_class,F1_per_class
    
def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre_p = classpre[1]
        recall_p = classrecall[1]
        iou_p = IU[1]
        F_score_p = 2*(recall_p*pre_p)/(recall_p+pre_p)

        pre_b = classpre[0]
        recall_b = classrecall[0]
        iou_b = IU[0]
        F_score_b = 2*(recall_b*pre_b)/(recall_b+pre_b)
    return globalacc, pre_p, recall_p, F_score_p, iou_p, pre_b, recall_b, F_score_b, iou_b
