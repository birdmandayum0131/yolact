import torch
import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time
from data import cfg
import pycocotools
from layers.box_utils import mask_iou

coco_ambiguous_class = np.array(range(81))
coco_ambiguous_class[[1,3]] = 1
coco_ambiguous_class[[2,5,7]] = 2
coco_ambiguous_class = coco_ambiguous_class.tolist()
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def greedy_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], torch.tensor(list(range(cost_matrix.shape[0]))), torch.tensor(list(range(cost_matrix.shape[1])))
    cost_matrix = torch.from_numpy(cost_matrix).cuda()
    matchMatrix_A = torch.zeros(cost_matrix.shape).cuda().long()
    matchMatrix_B = torch.zeros(cost_matrix.shape).cuda().long()
    _tmp_max_dist, _tmp_max_indices = torch.min(cost_matrix, dim = 1) #K_min_dist, K_min_indices
    matchMatrix_A[ range(matchMatrix_A.shape[0]) ,_tmp_max_indices] = 1 #
    _tmp_max_dist, _tmp_max_indices = torch.min(cost_matrix, dim = 0) #N_min_dist, N_min_indices
    matchMatrix_B[ _tmp_max_indices, range(matchMatrix_B.shape[1]) ] = 1#
    isOverThreshold = (cost_matrix < thresh).long()
    matchMatrix = ((matchMatrix_A + matchMatrix_B + isOverThreshold) == 3).long()
    
    _tmp_max_dist, _tmp_max_indices = torch.max(matchMatrix, dim = 1) # K_max_indices
    _tmp_sum = torch.sum(matchMatrix,dim=1) #K_sum_matrix
    _tmp_hasMatch = _tmp_sum > 0 #or == 1#K_hasMatch
    _tmp_noMatch = _tmp_sum == 0 #K_noMatch
    for ix in range(cost_matrix.shape[0]):
        if _tmp_hasMatch[ix]:
            matches.append([ix,_tmp_max_indices[ix]])
    unmatched_a = unmatched_a[_tmp_noMatch].tolist()
    _tmp_max_dist, _tmp_max_indices = torch.max(matchMatrix, dim = 0) # K_max_indices
    _tmp_sum = torch.sum(matchMatrix,dim=0) #K_sum_matrix
    _tmp_noMatch = _tmp_sum == 0 #K_noMatch
    unmatched_b = unmatched_b[_tmp_noMatch].tolist()
    
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious
    
def mask_ious(amasks, bmasks):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros( (len(amasks), len(bmasks)), dtype=np.float)
    if ious.size == 0:
        return ious
    
    amasksM = np.asarray([[mask]*len(bmasks) for mask in amasks])
    bmasksM = np.asarray([bmasks]*len(amasks))
    
    
    """
    amasks: [m1,n] m1 means number of predicted objects 
    bmasks: [m2,n] m2 means number of gt objects
    Note: n means image_w x image_h
    """
    intersection = np.sum(((amasksM + bmasksM)==2).astype(np.int), axis=(2,3))
    union = np.sum(((amasksM + bmasksM)>=1).astype(np.int), axis=(2,3))
    ious = intersection / union
    return ious
    '''
    return mask_iou(amasks,bmasks).cpu().numpy()
    '''


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    
    cost_matrix = 1 - _ious
    if cfg.match_cls_respective:
        if not (len(atracks)==0 or len(btracks)==0):
            a_classes = np.asarray([[coco_ambiguous_class[track.class_id]]*len(btracks) for track in atracks], dtype=np.int)
            b_classes = np.asarray([[coco_ambiguous_class[track.class_id] for track in btracks]]*len(atracks), dtype=np.int)
            cross_class_cost = (a_classes != b_classes).astype(np.float)
            cost_matrix = cost_matrix + cross_class_cost

    return cost_matrix

def mask_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        amasks = atracks
        bmasks = btracks
    else:
        amasks = [track.mask for track in atracks]
        bmasks = [track.mask for track in btracks]
    
    _ious = mask_ious(amasks, bmasks)
    #print(_ious)
    cost_matrix = 1 - _ious
    
    if cfg.match_cls_respective:
        if not (len(atracks)==0 or len(btracks)==0):
            a_classes = np.asarray([[coco_ambiguous_class[track.class_id]]*len(btracks) for track in atracks], dtype=np.int)
            b_classes = np.asarray([[coco_ambiguous_class[track.class_id] for track in btracks]]*len(atracks), dtype=np.int)
            cross_class_cost = (a_classes != b_classes).astype(np.float)
            cost_matrix = cost_matrix + cross_class_cost
    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.curr_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    if cfg.match_cls_respective:
        det_classes = np.asarray([[coco_ambiguous_class[track.class_id] for track in detections]]*track_features.shape[0], dtype=np.int)
        track_classes = np.asarray([[coco_ambiguous_class[track.class_id]]*det_features.shape[0] for track in tracks], dtype=np.int)
        cross_class_cost = (det_classes != track_classes).astype(np.float)*2
        cost_matrix = cost_matrix + cross_class_cost
    
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
    
def fuse_maskiou(cost_matrix, iou_cost_matrix, iou_threshold=0.5):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_cost_matrix = (iou_cost_matrix>iou_threshold).astype(np.float)*2
    cost_matrix = cost_matrix+iou_cost_matrix
    return cost_matrix
