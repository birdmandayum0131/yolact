import os
import numpy as np
import cv2
import torch
import math
from PIL import Image
from utils.augmentations import FastBaseTransform
from yolact import Yolact
from torch.utils.data import DataLoader
from coefTracker import coefPredictNet_v1
from layers.output_utils import reproduce_mask
import torch.nn.functional as F

save_folder = "CTweights/"
yolact_weight = 'weights/yolact_base_54_800000.pth'
default_davis_root = 'D:/Bird/DAVIS'
default_coef_path = 'train_coef/'

args = parser.parse_args()
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')

class DAVIScoefDataset(data.Dataset):
    def __init__(self, davis_root, coef_path):
        self.davis_root = davis_root#os.path.join(davis_root, 'Annotations_separate', '480p')
        self.coef_path = coef_path
        self.dataTupleList = self.getTupleList(coef_path)
        
    def getTupleList(self, coef_path):
        dataList = []
        for coefFile in os.listdir(coef_path):
            filePath = os.path.join(coef_path, coefFile)
            file = open(filePath, 'r')
            seqName = fileA.readline().rstrip('\n')
            _ = fileA.readline() #tracking ID
            gtID = int(fileA.readline())
            firstCoef = fileA.readline()
            firstCoef = np.array(firstCoef.split()[2:]).astype(float)
            secondCoef = fileA.readline()
            while secondCoef:
                secondCoef = secondCoef.split()
                frameID = int(secondCoef[0])
                secondCoef = np.array(secondCoef[2:]).astype(float)
                dataList.append((seqName, gtID, frameID, firstCoef, secondCoef))
                firstCoef = secondCoef
                secondCoef = fileA.readline()
        return dataList
        
    def __len__(self):
        return len(self.dataTupleList)
        
    def __getitem__(self, index):
        seqName, gtID, frameID, np1stcoef, np2ndcoef = self.dataTupleList[index]
        jpegImage = os.path.join(self.davis_root, 'JPEGImages', '480p', seqName, '%05d.jpg'%(frameID))
        jpegImage = torch.from_numpy(cv2.imread(jpegImage)).cuda().float()
        
        gtImage = os.path.join(self.davis_root, 'Annotations_separate', '480p', seqName, str(gtID), '%05d.png'%(frameID))
        gtImage = np.array(Image.open(gtImage))
        
        '''use gpu to calc ground truth bbox'''
        gtImage = torch.tensor(gtImage).cuda()
        rowSum = torch.sum(gtImage,dim=1)
        colSum = torch.sum(gtImage,dim=0)
        _ , x2 = torch.max((colSum > 0) * torch.tensor(range(colSum.shape[0])), dim=0)
        _ , x1 = torch.max((colSum > 0) * (colSum.shape[0] - torch.tensor(range(colSum.shape[0]))), dim=0)
        _ , y2 = torch.max((rowSum > 0) * torch.tensor(range(rowSum.shape[0])), dim=0)
        _ , y1 = torch.max((rowSum > 0) * (rowSum.shape[0] - torch.tensor(range(rowSum.shape[0]))), dim=0)
        boxes = torch.cat((x1,y1,x2,y2), dim=0).transpose(0,1)
        
        return jpegImage, (gtImage, boxes, np1stcoef, np2ndcoef)

def simpleBCEloss(w, h, gt_box, proto_data, coef, gt_mask):
    box_tmp = gt_box.clone().float()
    box_tmp[:,0] /= w
    box_tmp[:,2] /= w
    box_tmp[:,1] /= h
    box_tmp[:,3] /= h
    pred_masks = proto_data @ coef.t()
    pred_masks = cfg.mask_proto_mask_activation(pred_masks)
    pred_masks = F.interpolate(pred_masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
    loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), torch.clamp(mask_t, 0, 1), reduction='none')
    return loss
    
def train():
    '''建立存檔資料夾'''
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    dataset = DAVIScoefDataset(default_davis_root, default_coef_path)
    net = Yolact()
    net.load_weights(yolact_weight)
    net.eval()
    net = net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = True
    
    coefNet = coefPredictNet_v1()
    coefNet = coefNet.cuda()
    coefNet.train()
    
    optimizer = optim.SGD(coefPredictNet_v1.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay)
    
    criterion
    
    max_iter = 10000
    iteration = 0
    epoch_size = len(dataset)
    num_epochs = math.ceil(max_iter / epoch_size)
    
    data_loader = DataLoader(dataset, batch_size=1)
    save_path = "coefNet_"
    
    
    print('Begin training!')
    print()
    for epoch in range(num_epochs):
        for datum in data_loader:
            if iteration == cfg.max_iter:
                break
            iteration += 1
            optimizer.zero_grad()
            jpegImage, gtImage, gtbox, np1stcoef, np2ndcoef = datum
            ###############################################
            import pdb
            pdb.set_trace()
            jpegImage = FastBaseTransform()(jpegImage.unsqueeze(0))
            dets = net(jpegImage)
            proto = dets[0]['proto']
            coef1 = torch.from_numpy(np1stcoef).unsqueeze(0).cuda()
            coef2 = torch.from_numpy(np2ndcoef).unsqueeze(0).cuda()
            refineCoef = coefNet(coef1, coef2)
            _ , new_mask = simpleBCEloss(gtImage.shape[1], gtImage.shape[0], gtbox, proto, refineCoef, gtbox.unsqueeze(2))
            
            
            
if __name__ == '__main__':
    train()
    
    