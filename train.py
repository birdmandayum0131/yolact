from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int, #default batch size as 8
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, #resume training from checkpoint file
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int, 
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        '''根據batch size為8的倍數調整learning rate, iteration, steps'''
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

'''檢查有無gpu'''
if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)
'''
若每個gpu分配到的local batch size < 6
則不適合做batch normalization
'''
if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True
'''這啥? 百事可樂?'''
loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

'''看看你有沒有偷藏你的gpu不用'''
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

'''
用來算loss的網路
優化平行運算用(?)
因為nn.DataParallel必須輸入nn.module
所以把loss用網路包著(?)
'''
class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses
'''
yolact自己做的平行運算
對他們的data來說效果比pytorch原生的更好
'''
class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """
    
    '''
    把data平行分開來做batch normalization
    同時做資料分散跟資料準備(?)
    func:prepare_data之後再看
    '''
    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)
    
    '''把平行運算後的data聚集回來'''
    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

'''training主程式'''
def train():
    '''建立存檔資料夾'''
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    '''預設用COCO訓練'''
    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    '''是否做validation'''
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))
    '''
    平行包成兩個一樣的網路
    不確定用意
    saving跟loading不要比較好(?)
    '''
    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()
    
    '''log資訊'''
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)
    
    '''
    避免平行運算出bug
    而且yolact在training時沒有使用timer
    保險起見關閉timer
    '''
    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()
    
    '''resume training的設定'''
    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)
    
    '''大家都用SGD'''
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)
    '''多gpu的local batch size設定'''
    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)
    
    '''平行運算(loss網路(yolact))'''
    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    '''
    初始化
    流程 = 跑一遍全黑影像初始化(?)
    但因為全黑影像會破壞mean
    因此不管怎樣都要在初始化前freeze_bn
    #Note
    freeze_bn(True)為解除freeze_bn
    yolact.train()會自動freeze_bn
    '''
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    '''沒有包在if裡面'''
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    '''1 epoch = ? iterations'''
    epoch_size = len(dataset) // args.batch_size
    '''ceiling向上取整'''
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    '''每次step是用來調整learning rate'''
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0
    '''data loader'''
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    '''
    定義func:save_path
    根據epoch以及iteration產生string:path
    '''
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    '''
    動態平均值產生器(?)
    可以回傳固定前面n個stack的平均值
    default n = 1000
    '''
    time_avg = MovingAverage()
    
    '''
    哪來那麼多Loss(?)
    n = 100
    '''
    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print()
    '''
    begin training
    讓你可以用ctrl+c做interrupt的training寫法
    '''
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            '''
            iteration已經被start_iter覆蓋過
            在start_iter之前都維持continue
            執行完後當前剩餘iteration(start_iter-current_iter)會小於一個epoch_size
            '''
            if (epoch+1)*epoch_size < iteration:
                continue
            '''開始拉資料(datum?)'''
            for datum in data_loader:
                '''--------------------------------------------setting--------------------------------------------'''
                # Stop if we've reached an epoch if we're resuming from start_iter
                '''
                跟上面一樣
                若剛好做完一個epoch
                就把training部分跳過
                檢查最後做過的epoch的validation情況
                '''
                if iteration == (epoch+1)*epoch_size:
                    break
                '''
                若iteration reach到max_iter一樣處理
                '''
                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break
                '''
                # A list of settings to apply after the specified iteration. Each element of the list should look like
                # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
                用來在指定iteration改變config的寫法
                記得指定改變的config_dict都要初始化過(key, value)
                如果改變不少的話應該要換個寫法(?)(solved)
                    下面一行就移除多於的設定了-_-
                '''
                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        '''config改變了 loss重新統計比較make sense(?)'''
                        for avg in loss_avgs:
                            avg.reset()
                '''移除已執行過的設定'''
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]
                '''
                learning rate的warm up機制(什麼怪機制)
                如果有設定過
                則利用linear interpolation做warmup_init~learning rate的漸進
                '''
                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)
                '''
                假如還有步數要走&&下一步的iteration已經到/過了
                那就往下走一步
                更新learning rate
                就算resume也可以正常更改(robust)
                '''
                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                '''--------------------------------------------setting--------------------------------------------'''
                '''--------------------------------------------training--------------------------------------------'''
                '''其實training的步驟也真的只有這樣'''
                '''step 1:zero gradient'''
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()
                '''
                step 2:output = network.forward(data)
                step 3:compute loss
                ------>loss = DataParallelModule( LossNetwork( ForwardNetwork(data)))
                #寫成loss network以及data parallel module後
                #看起來非常漂亮
                '''
                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                '''
                step 3:compute loss
                因為dataparallel寫法
                所以從網路output出來後再一起mean()
                重新放到字典裡
                再利用list comprehension計算總loss(loss權重呢?)
                #inf = infinite = 正/負無限大 or NaN
                #他有寫一個no_inf_mean
                #只計算有限數值(no_inf)的mean值
                '''
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])
                '''
                step 4:backpropagation
                不管loss是否無限大都backward
                先釋放Video RAM再說
                反正只要optimizer不step都沒事(?)
                '''
                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                '''--------------------------------------------training--------------------------------------------'''
                '''計算loss的動態平均值(包括infinte value)'''
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())
                '''計算本次iteration的執行時間'''
                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time
                '''
                第一個iteration包含了setup的時間
                不列入計算
                '''
                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)
                '''每10個iteration顯示一次'''
                if iteration % 10 == 0:
                    '''format:eta_str = hr:min:sec(.後的毫秒被split掉了)'''
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    '''loss加總成total loss'''
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    '''
                    將loss整理成一個list:[key, value, key, value]形式
                    #內制函數sum會根據初始值來決定加總方式
                    #初始值default=0
                    #若設定為空陣列即可用來做陣列合併
                    '''
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    '''顯示資訊'''
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                '''log資訊到log檔中'''
                if args.log:
                    '''四捨五入的小數點精確位'''
                    precision = 5
                    '''全部取到第5位'''
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    '''增加一個T來放total loss'''
                    loss_info['T'] = round(loss.item(), precision)
                    '''很慢 不要隨便用'''
                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                    '''log主程式'''    
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                '''該iteration正式結束'''
                iteration += 1
                '''
                每save_interval個iteration就做一個checkpoint
                start_iter就不用多存一次了
                '''
                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    '''
                    只保留最後一個checkpoint的選項
                    要隨時把舊的checkpoint刪除
                    所以把檔名先記起來
                    應該會獲得"上一個檔案"的名字
                    '''
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)
                    '''顯示iteration'''
                    print('Saving state, iter:', iteration)
                    '''做checkpoint'''
                    yolact_net.save_weights(save_path(epoch, iteration))
                    '''
                    若有開啟只存最後的checkpoint
                    而且上一個檔名是存在的話
                    代表有舊檔案需要被刪除了
                    '''
                    if args.keep_latest and latest is not None:
                        '''
                        若keep_latest_interval <= 0代表真的只留最後的檔案
                        若還有值代表間格keep_latest_interval個iteration還是會存一次的
                        '''
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            '''一次iteration正式結束'''
            '''每個epoch都會檢查的部分'''
            '''若validation_epoch存在'''
            # This is done per epoch
            if args.validation_epoch > 0:
                '''若非第一個epoch且是validation_epoch'''
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    '''做validation'''
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        '''training結束時再做一次validation'''
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        '''此段可以讓ctrl+c儲存iterrupt的model'''
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()
    '''training結束時再存一次'''
    yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
