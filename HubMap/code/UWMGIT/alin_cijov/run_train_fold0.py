import os
import cv2
import pdb
import time
import random

import torch
from torch import nn
import torch.cuda.amp as amp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler
import torch.nn.functional as F
from torchmetrics.functional import dice_score
from torch.optim.lr_scheduler import StepLR

is_amp = True
import logging
import pandas as pd
from sklearn.model_selection import KFold

import numpy as np
from itertools import repeat
import collections.abc

import warnings
warnings.filterwarnings('ignore')

from dataset import *
from augmentation import *
from model import *

############################################################
####### Validation
############################################################
def validate(net, valid_loader):
    
    valid_num = 0
    valid_probability = []
    valid_mask = []
    valid_loss = 0

    net = net.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):

                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask' ] = batch['mask' ].cuda()
                batch['organ'] = batch['organ'].cuda()

                output = net(batch)
                loss0  = output['bce_loss'].mean()

        valid_probability.append(output['probability'].data.cpu().numpy())
        valid_mask.append(batch['mask'].data.cpu().numpy())
        
        valid_num += batch_size
        valid_loss += batch_size*loss0.item()

        #debug
        if 0 :
            pass
            organ = batch['organ'].data.cpu().numpy()
            image = batch['image']
            mask  = batch['mask']
            probability  = output['probability']

            for b in range(batch_size):
                m = tensor_to_image(image[b])
                t = tensor_to_mask(mask[b,0])
                p = tensor_to_mask(probability[b,0])
                overlay = result_to_overlay(m, t, p )

                text = label_to_organ[organ[b]]
                draw_shadow_text(overlay,text,(5,15),0.7,(1,1,1),1)

                image_show_norm('overlay',overlay,min=0,max=1,resize=1)
                cv2.waitKey(0)

        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),(time.time() - start_timer)),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))

    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    loss = valid_loss/valid_num

    dice = compute_dice_score(probability, mask)
    dice = dice.mean()
    
    return [dice, loss,  0, 0]


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

############################################################
####### Computetion Metric
############################################################
def compute_dice_score(probability, mask):
    N = len(probability)
    p = probability.reshape(N,-1)
    t = mask.reshape(N,-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum(-1) + t.sum(-1)
    overlap = (p*t).sum(-1)
    dice = 2*overlap/(uion+0.0001)
    return dice


def run_train(fold):
    
    ###################################
    ####### hyper-paramter
    ###################################

    fold = fold
    out_dir = './result/upernet-swin-v1-tiny-aux5-768/fold-%d' % (fold)
    initial_checkpoint = None

    start_lr   = 5e-5 #0.0001
    batch_size = 8 #32 #32

    for f in ['checkpoint','train','valid','backup'] : 
        os.makedirs(out_dir +'/'+f, exist_ok=True)

    log = open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % ('Swin', '-' * 64))
    log.write('\n')

    log.write('** dataset setting **\n')
    train_df, valid_df = make_fold(fold)
    train_dataset = HubmapDataset(train_df, train_augment5b)
    valid_dataset = HubmapDataset(valid_df, valid_augment5)

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 8, # if dubug 0
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = False,
        collate_fn = null_collate,
    )


    log.write('fold = %s\n'%str(fold))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')

    scaler = amp.GradScaler(enabled = is_amp)
    net = Net().cuda()

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=False)  #True
    else:
        start_iteration = 0
        start_epoch = 0
        net.load_pretrain() # pretrain


    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')


    ## optimiser ----------------------------------
    if 0: ##freeze
        for p in net.stem.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                
    #freeze_bn(net)

    #-----------------------------------------------

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('\n')

    num_iteration = 1000*len(train_loader)
    iter_log   = len(train_loader)*3 #479
    iter_valid = iter_log
    iter_save  = iter_log

    ############################################################
    ####### Training
    ############################################################
    log.write('** start training here! **\n')
    log.write('   batch_size = %d \n'%(batch_size))
    log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
    log.write('rate     iter  epoch | dice   loss   tp     tn     | loss           | time           \n')
    log.write('-------------------------------------------------------------------------------------\n')

    def message(mode='print'):
        asterisk = ' '
        if mode==('print'):
            loss = batch_loss
        if mode==('log'):
            loss = train_loss
            if (iteration % iter_save == 0): asterisk = '*'

        text = \
            ('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.3f  %4.3f  %4.4f  %4.3f   | '%(*valid_loss,) + \
            '%4.3f  %4.3f   | '%(*loss,) + \
            '%s' % ((time.time() - start_timer))

        return text

    valid_loss = np.zeros(4,np.float32)
    train_loss = np.zeros(2,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = time.time()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0

    while iteration < num_iteration:
        for t, batch in enumerate(train_loader):

            if iteration%iter_save==0:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d.model.pth' %  (iteration))
                    pass


            if (iteration%iter_valid==0):
                valid_loss = validate(net, valid_loader)
                pass


            if (iteration%iter_log==0) or (iteration%iter_valid==0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')
                
            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].half().cuda()
            batch['mask' ] = batch['mask' ].half().cuda()
            batch['organ'] = batch['organ'].cuda()


            net.train()
            net.output_type = ['loss']
            if 1:
                with amp.autocast(enabled = is_amp):
                    pdb.set_trace()
                    output = net(batch)
                    loss0  = output['bce_loss'].mean()
                    loss1  = output['aux2_loss'].mean()

                optimizer.zero_grad()
                scaler.scale(loss0+0.2*loss1).backward()

                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()


            # print statistics  --------
            batch_loss[:2] = [loss0.item(),loss1.item()]
            sum_train_loss += batch_loss
            sum_train += 1
            if t % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
            epoch += 1 / len(train_loader)
            iteration += 1
            
        torch.cuda.empty_cache()
        
    log.write('\n')
    log.close()
    

if __name__ == '__main__':
    run_train(fold=0) # command、windows
