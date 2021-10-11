import os
import numpy as np
import cv2
import torch
import math
from data import cfg
from PIL import Image
from utils.augmentations import FastBaseTransform
from yolact import Yolact
import torch.utils.data as data
from torch.utils.data import DataLoader
from coefTracker import *
from layers.output_utils import reproduce_mask
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch.optim as optim
import pdb
import time
save_folder = "CTweights/V9_AdamW_cosine_v4set_extend/"
yolact_weight = 'weights/yolact_plus_base_54_800000.pth'
default_davis_root = 'D:/Bird/DAVIS'
default_coef_path = 'coef_traing_set_v4/'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None, type=str, #resume training from checkpoint file
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=0, type=int, 
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
args = parser.parse_args()

class coefTripletDataset(data.Dataset):
    def __init__(self, davis_root, coef_path):
        self.davis_root = davis_root#os.path.join(davis_root, 'Annotations_separate', '480p')
        self.coef_path = coef_path
        self.dataTripletList = self.dataTripletList(coef_path)
        
    def dataTripletList(self, coef_path):
        dataList = []
        for coefFile in os.listdir(coef_path):
            filePath = os.path.join(coef_path, coefFile)
            file = open(filePath, 'r')
            seqName = file.readline().rstrip('\n')
            _ = file.readline() #tracking ID
            gtID = int(file.readline())
            firstrow = file.readline().split()
            secondrow = file.readline().split()
            thirdrow = file.readline()
            while thirdrow:
                thirdrow = thirdrow.split()
                
                firstCoef = np.array(firstrow[2:]).astype(float)
                secondCoef = np.array(secondrow[2:]).astype(float)
                thirdCoef = np.array(thirdrow[2:]).astype(float)
                
                firstframe = int(firstrow[0])
                thirdframe = int(thirdrow[0])
                
                firstConf = float(firstrow[1])
                secondConf = float(secondrow[1])
                
                if (thirdframe-firstframe)==2:
                    dataList.append((seqName, gtID, (firstframe+1), firstCoef, secondCoef, thirdCoef, firstConf, secondConf))
                firstrow = secondrow
                secondrow = thirdrow
                thirdrow = file.readline()
        return dataList
        
    def __len__(self):
        return len(self.dataTripletList)
        
    def __getitem__(self, index):
        seqName, gtID, frameID, np1stcoef, np2ndcoef, np3rdcoef, firstConf, secondConf= self.dataTripletList[index]
        #jpegImage = os.path.join(self.davis_root, 'JPEGImages', '480p', seqName, '%05d.jpg'%(frameID))
        #jpegImage = torch.from_numpy(cv2.imread(jpegImage)).cuda().float()
        
        #gtImage = os.path.join(self.davis_root, 'Annotations_separate', '480p', seqName, str(gtID), '%05d.png'%(frameID))
        #gtImage = np.array(Image.open(gtImage))
        
        '''use gpu to calc ground truth bbox'''
        #gtImage = torch.tensor(gtImage).cuda()
        #rowSum = torch.sum(gtImage,dim=1)
        #colSum = torch.sum(gtImage,dim=0)
        #_ , x2 = torch.max((colSum > 0) * torch.tensor(range(colSum.shape[0])).cuda(), dim=0)
        #_ , x1 = torch.max((colSum > 0) * (colSum.shape[0] - torch.tensor(range(colSum.shape[0])).cuda()), dim=0)
        #_ , y2 = torch.max((rowSum > 0) * torch.tensor(range(rowSum.shape[0])).cuda(), dim=0)
        #_ , y1 = torch.max((rowSum > 0) * (rowSum.shape[0] - torch.tensor(range(rowSum.shape[0])).cuda()), dim=0)
        #boxes = torch.cat((x1.unsqueeze(0),y1.unsqueeze(0),x2.unsqueeze(0),y2.unsqueeze(0)), dim=0)
        
        return np1stcoef, np2ndcoef, np3rdcoef, firstConf, secondConf, seqName
def parsemodelNameInfo(path:str):
    file_name = os.path.basename(path)
    
    if file_name.endswith('.pth'):
        file_name = file_name[:-4]
    
    params = file_name.split('_')
        
    #model_name = '_'.join(params[:-2])
    epoch = params[-3]
    iteration = params[-2]
    last_loss = params[-1]
    
    return int(epoch), int(iteration), float(last_loss)

def simpleBCEloss(w, h, gt_box, proto_data, coef, gt_mask):
    box_tmp = gt_box.clone().float()
    box_tmp[:,0] /= w
    box_tmp[:,2] /= w
    box_tmp[:,1] /= h
    box_tmp[:,3] /= h
    
    pred_masks = proto_data @ coef.t()
    pred_masks = cfg.mask_proto_mask_activation(pred_masks)
    pred_masks = pred_masks.permute(2, 0, 1).contiguous()
    pred_masks = F.interpolate(pred_masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
    pred_masks = pred_masks.permute(1, 2, 0).contiguous()
    
    loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), torch.clamp(gt_mask, 0, 1), reduction='none')
    return loss.sum()/w/h
    
def train():
    '''建立存檔資料夾'''
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    last_loss = 0
    iteration = 0
    
    last_time = time.time()
    dataset = coefTripletDataset(default_davis_root, default_coef_path)
    '''
    net = Yolact()
    net.load_weights(yolact_weight)
    net.eval()
    net = net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = True
    '''
    coefNet = coefPredictNet_v9()
    coefNet = coefNet.cuda()
    coefNet.train()
    
    
    if args.resume:
        last_epoch, iteration, last_loss = parsemodelNameInfo(args.resume)
        coefNet.load_weights(args.resume)
    
    optimizer = optim.AdamW(coefNet.parameters(), lr=0.001)
    
    max_iter = 1000000
    
    epoch_size = len(dataset)
    num_epochs = math.ceil(max_iter / epoch_size)
    
    data_loader = DataLoader(dataset, batch_size=1)
    fileformat = "coefNetV9_%d_%d_%.3f.pth" #epoch, iteration, loss
    
    print('epoch_size : ' + str(epoch_size))
    print('Begin training!')
    print()
    
    try:
        loss = torch.zeros(1)
        totalLoss = 0
        for epoch in range(num_epochs):
            last_loss = totalLoss
            totalLoss = 0
            seqName_prev = None
            if (epoch+1)*epoch_size < iteration:
                continue
            for datum in data_loader:
                #with torch.autograd.set_detect_anomaly(True):
                if iteration == max_iter:
                    break
                
                optimizer.zero_grad()
                coef1, coef2, coef3, conf1, conf2, seqName_crnt = datum
                
                if not seqName_crnt == seqName_prev:
                    coefNet.init_hidden(batch_size=1)
                seqName_prev = seqName_crnt
                #jpegImage = FastBaseTransform()(jpegImage)
                
                #dets = net(jpegImage)
                #proto = dets[0]['proto']
                coef1 = coef1.cuda().float()
                coef2 = coef2.cuda().float()
                coef3 = coef3.cuda().float()
                predictCoef = coefNet(coef1, coef2)
                loss = 1 - F.cosine_similarity(predictCoef, coef3)
                #loss = torch.cdist(predictCoef, coef3)
                #loss = torch.cdist(coef2, coef3)
                
                
                #loss = simpleBCEloss(gtImage.shape[2], gtImage.shape[1], gtbox, proto, refineCoef, gtImage.permute(1, 2, 0).contiguous())
                loss.backward(retain_graph=True)
                if torch.isfinite(loss).item():
                    optimizer.step()
                    
                iteration += 1
                totalLoss += loss.item()
                
                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    save_path = fileformat%(epoch, iteration, last_loss)
                    coefNet.save_weights(os.path.join(save_folder, save_path))
            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time
            print("loss : %.3f || timer : %.3f"%(totalLoss, elapsed))
    except KeyboardInterrupt:
        save_path = fileformat%(epoch, iteration, last_loss)
        coefNet.save_weights(os.path.join(save_folder, save_path))
        
if __name__ == '__main__':
    train()
    
    