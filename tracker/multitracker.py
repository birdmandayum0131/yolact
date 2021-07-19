import numpy as np
from numba import jit
from collections import deque
from tracking_utils.kalman_filter import KalmanFilter as FairMoT_Kalman
from tracker import matching
from .basetrack import BaseTrack, TrackState
import filterpy.kalman as fpy_kalman
from filterpy.common import Q_discrete_white_noise
from data import cfg
from layers.output_utils import reproduce_mask
import torch


class STrack(BaseTrack):
    shared_kalman = FairMoT_Kalman()
    def __init__(self, class_id, score, tlwh, mask, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.smooth_coef, self.coef_covariance = None, None
        self.is_activated = False
        self.last_coef = temp_feat
        
        self.score = score
        self.tracklet_len = 0
        self.mask = mask
        self.class_id = class_id
        
        self.curr_feat = temp_feat
        #self.update_features(temp_feat)
        #self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.8


    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        self.features.append(feat)
        '''I think i dont need this in this repo'''
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    def predict_coef(self):
        '''
        if cfg.use_score_confidence:
            Q = Q * (1 - self.score) * self.alpha
        '''
        self.coef_x, self.coef_covariance = fpy_kalman.predict(self.coef_x, self.coef_covariance, self.F, self.Q)
        self.curr_feat = self.coef_x[:32]
        

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = 0#self.next_id()
        if cfg.use_bbox_kalman:
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
            
        if cfg.use_coef_kalman and cfg.linear_coef:
            if cfg.use_coef_motion_model:
                self.F = np.eye(2 * 32) #motion matrix
                for i in range(32):
                    self.F[i, 32 + i] = 1
                self.Q = np.eye(32*2) * 0.15*0.15 #Process uncertainty/noise
                coef_vel = np.zeros_like(self.curr_feat)
                self.coef_x = np.r_[self.curr_feat, coef_vel]
                self.coef_covariance = np.eye(32*2)
                self.coef_covariance[:32,:] *= 0.15*0.15
                self.coef_covariance[32:,:] *= 0.05*0.05
                self.H = np.zeros((32,2*32)) #measurement function
                for i in range(32):
                    self.H[i,i] = 1
            else:
                self.F = np.eye(32) #transistion matrix
                self.Q = np.eye(32) * 0.15*0.15 #Process uncertainty/noise
                self.coef_x = self.curr_feat
                self.coef_covariance =  np.eye(32)*0.15*0.15
                self.H = np.eye(32) #measurement function
            
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 0:
            self.is_activated = True
            self.track_id = self.next_id()
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, protos, new_id=False):
        new_tlwh = new_track.tlwh
        #self.update_features(new_track.curr_feat)
        self.mask = new_track.mask
        if cfg.linear_coef:
            if cfg.use_coef_kalman:
                if cfg.use_coef_motion_model:
                    R = np.eye(32) * 0.3*0.3#measurement uncertainty/noise
                else:
                    R = np.eye(32) #measurement uncertainty/noise
                    '''
                    if cfg.use_score_confidence:
                        R = R * (1 - new_track.score) * self.alpha
                    '''
                self.coef_x, self.coef_covariance, y, K, S, log_likelihood = fpy_kalman.update(self.coef_x, self.coef_covariance, new_track.curr_feat, R, self.H, return_all=True)
                self.curr_feat = self.coef_x[:32]
            else:
                self.curr_feat = self.curr_feat * self.alpha + new_track.curr_feat * (1 - self.alpha)
            boxes, masks = reproduce_mask(self.mask.shape[1], self.mask.shape[0], torch.tensor(new_track.tlbr).unsqueeze(0).cuda(), torch.tensor(self.curr_feat).unsqueeze(0).cuda(), protos)
            boxes = boxes.float().squeeze(0).cpu().numpy()
            new_tlwh = self.tlbr_to_tlwh(boxes)
            if (new_tlwh[2] == 0) or (new_tlwh[3] == 0):
                return 1
            
            self.mask = masks.squeeze(0).cpu().numpy()
        else:
            self.curr_feat = new_track.curr_feat
        
        if cfg.use_bbox_kalman:
            self.mean, self.covariance = self.kalman_filter.update( self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh) )
        self.last_coef = new_track.last_coef
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        
        self.class_id = new_track.class_id
        self._tlwh = new_tlwh
        return 0

    def update(self, new_track, frame_id, protos, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        
        #if update_feature:
            #self.update_features(new_track.curr_feat
        self.mask = new_track.mask
        if cfg.linear_coef:
            if cfg.use_coef_kalman:
                if cfg.use_coef_motion_model:
                    R = np.eye(32) * 0.3*0.3#measurement uncertainty/noise
                else:
                    R = np.eye(32) #measurement uncertainty/noise
                    '''
                    if cfg.use_score_confidence:
                        R = R * (1 - new_track.score) * self.alpha
                    '''
                self.coef_x, self.coef_covariance, y, K, S, log_likelihood = fpy_kalman.update(self.coef_x, self.coef_covariance, new_track.curr_feat, R, self.H, return_all=True)
                self.curr_feat = self.coef_x[:32]
            else:
                self.curr_feat = self.curr_feat * self.alpha + new_track.curr_feat * (1 - self.alpha)
            boxes, masks = reproduce_mask(self.mask.shape[1], self.mask.shape[0], torch.tensor(new_track.tlbr).unsqueeze(0).cuda(), torch.tensor(self.curr_feat).unsqueeze(0).cuda(), protos)
            boxes = boxes.float().squeeze(0).cpu().numpy()
            new_tlwh = self.tlbr_to_tlwh(boxes)
            if (new_tlwh[2] == 0) or (new_tlwh[3] == 0):
                return 1
            
            self.mask = masks.squeeze(0).cpu().numpy()
        else:
            self.curr_feat = new_track.curr_feat
        self.last_coef = new_track.last_coef
        if cfg.use_bbox_kalman:
            self.mean, self.covariance = self.kalman_filter.update( self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        if self.tracklet_len == 3 and self.is_activated != True:
            self.is_activated = True
            self.track_id = self.next_id()
        
        self.score = new_track.score

        
        self.class_id = new_track.class_id
        self._tlwh = new_tlwh
        return 0
    
    def predict_mask(self, protos):
        boxes, masks = reproduce_mask(self.mask.shape[1], self.mask.shape[0], torch.tensor(self.tlbr).unsqueeze(0).cuda(), torch.tensor(self.curr_feat).unsqueeze(0).cuda(), protos)
        boxes = boxes.float().squeeze(0).cpu().numpy()
        new_tlwh = self.tlbr_to_tlwh(boxes)
        if (new_tlwh[2] == 0) or (new_tlwh[3] == 0):
            return 1
        self.mask = masks.squeeze(0).cpu().numpy()
        self._tlwh = new_tlwh
        self.score = 0
        return 0

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None or not cfg.use_bbox_kalman:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb