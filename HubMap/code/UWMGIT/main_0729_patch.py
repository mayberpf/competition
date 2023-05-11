###############################################################
##### @Title:  HuBMAP baseline
##### @Time:  2022/07/29
##### @Author: frank
##### @Struct: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @Describe: 
        # The Devastator - "hubmap-2022-256x256" 作者
        # 
##### @Reference:
        # [Training] - FastAI Baseline: https://www.kaggle.com/code/thedevastator/training-fastai-baseline
        # [Inference] - FastAI Baselin: https://www.kaggle.com/code/thedevastator/inference-fastai-baseline
###############################################################
import os
import pdb
import cv2
import time
import glob
import random

import rasterio
from rasterio.windows import Window
import tifffile

from cv2 import transform
import numpy as np
import pandas as pd
from tqdm import tqdm

from fastai.vision.all import *

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html

from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import albumentations as A # Augmentations

import segmentation_models_pytorch as smp # smp
# 从这里下载：https://github.com/Shujun-He/Hubmap-3rd-place-solution/
from resnext_unet import *

def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################
##### part0: data preprocess
###############################################################
# functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

#https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
#with transposed mask
def rle_encode_less_memory(img):
    #the image should be transposed
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def plot_visual(image, mask, pred, fold, idx, cmap):
    
    image = ((image.transpose(1,2,0)*std + mean)*255.0).astype(np.uint8)
    mask = mask.squeeze()
    pred = pred.squeeze()
    
    plt.figure(figsize=(16, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(image, vmin=0, vmax=255)
    plt.title("image", fontsize=10)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image, vmin=0, vmax=255)
    plt.imshow(mask, cmap=cmap, alpha=0.5)
    plt.title(f"mask", fontsize=10)    
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(image, vmin=0, vmax=255)
    plt.imshow(pred, cmap=cmap, alpha=0.5)
    plt.title(f"pred", fontsize=10)    
    plt.axis("off")

    plt.savefig(f"./visual/{fold}/{idx}.png")
    plt.close()
    
###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                # IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms


# https://www.kaggle.com/datasets/thedevastator/hubmap-2022-256x256
mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class build_dataset(Dataset):
    def __init__(self, df=None, label=True, transforms=None, CFG=None, idx = None, sz=None, reduce=reduce):
        self.label = label
        if self.label:
            ###########################################
            ##### >>>>>>> Use "hubmap-2022-256x256" Dataset <<<<<<
            ############################################
            self.df = df
            ids = df.id.astype(str).values
            self.third_data_train = os.path.join(CFG.third_data_path, "train")
            self.third_data_mask = os.path.join(CFG.third_data_path, "masks")
            self.file_names = [file_name for file_name in os.listdir(self.third_data_train) if file_name.split('_')[0] in ids]
            self.organ_to_label = {'kidney' : 0, 'prostate' : 1, 'largeintestine' : 2, 'spleen' : 3, 'lung' : 4}        
            self.label = label
            self.transforms = transforms
        else:
            ###########################################
            ##### >>>>>>> Use Original Dataset <<<<<<
            ############################################            
            self.original_data_train = os.path.join(CFG.data_path, "test_images")
            self.data = rasterio.open(os.path.join(self.original_data_train, idx+'.tiff'), transform = identity,
                                 num_threads='all_cpus')
            # some images have issues with their format 
            # and must be saved correctly before reading with rasterio
            if self.data.count != 3:
                subdatasets = self.data.subdatasets
                self.layers = []
                if len(subdatasets) > 0:
                    for i, subdataset in enumerate(subdatasets, 0):
                        self.layers.append(rasterio.open(subdataset))
            self.shape = self.data.shape
            self.reduce = reduce
            self.sz = reduce*sz
            self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
            self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
            self.n0max = (self.shape[0] + self.pad0)//self.sz
            self.n1max = (self.shape[1] + self.pad1)//self.sz
        
            
    def __len__(self):
        if self.label:
            return len(self.df)
        else:
            return self.n0max*self.n1max
    
    def __getitem__(self, idx):
        
        if self.label:
            file_name = self.file_names[idx]
            img = cv2.cvtColor(cv2.imread(os.path.join(self.third_data_train, file_name)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.third_data_mask, file_name),cv2.IMREAD_GRAYSCALE)
            
            if self.transforms:
                data = self.transforms(image=img, mask=mask)
                img  = data['image']
                mask  = data['mask']
            
            return img2tensor((img/255.0 - mean)/std), img2tensor(mask)
        else:
            # the code below may be a little bit difficult to understand,
            # but the thing it does is mapping the original image to
            # tiles created with adding padding, as done in
            # https://www.kaggle.com/iafoss/256x256-images ,
            # and then the tiles are loaded with rasterio
            # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
            n0,n1 = idx//self.n1max, idx%self.n1max
            # x0,y0 - are the coordinates of the lower left corner of the tile in the image
            # negative numbers correspond to padding (which must not be loaded)
            x0,y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz
            # make sure that the region to read is within the image
            p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
            p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
            img = np.zeros((self.sz,self.sz,3),np.uint8)
            # mapping the loade region to the tile
            if self.data.count == 3:
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1,2,3],
                    window=Window.from_slices((p00,p01),(p10,p11))), 0, -1)
            else:
                for i,layer in enumerate(self.layers):
                    img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i] =\
                    layer.read(1,window=Window.from_slices((p00,p01),(p10,p11)))
            
            if self.reduce != 1:
                img = cv2.resize(img,(self.sz//reduce,self.sz//reduce),
                                interpolation = cv2.INTER_AREA)
            #check for empty imges
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            if (s>s_th).sum() <= p_th or img.sum() <= p_th:
                #images with -1 will be skipped
                return img2tensor((img/255.0 - mean)/std), -1
            else: return img2tensor((img/255.0 - mean)/std), idx
        
        

def build_dataset_dataloader(df, fold, data_transforms, CFG):
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    train_dataset = build_dataset(df = train_df, label=True, transforms=data_transforms['train'], CFG=CFG)
    valid_dataset = build_dataset(df = valid_df, label=True, transforms=data_transforms['valid_test'], CFG=CFG)
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=CFG.num_worker, 
                              shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=CFG.num_worker, 
                              shuffle=False, pin_memory=True)
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://smp.readthedocs.io/en/latest/encoders_timm.html
def build_model(CFG, test_flag=False):
    if test_flag:
        pretrain_weights = None
    else:
        pretrain_weights = "imagenet"
    # model = smp.Unet(
    #         encoder_name=CFG.backbone,
    #         encoder_weights=pretrain_weights, 
    #         in_channels=3,             
    #         classes=CFG.num_classes,   
    #         activation=None,
    #     )
    model = UneXt50()
    model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    # BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    BCELoss     = nn.BCEWithLogitsLoss()
    DiceLoss    = smp.losses.DiceLoss(mode='binary')
    return {"BCELoss":BCELoss, "DiceLoss":DiceLoss}

###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
# def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
#     y_true = y_true.to(torch.float32)
#     y_pred = (y_pred>thr).to(torch.float32)
#     inter = (y_true*y_pred).sum(dim=dim)
#     den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
#     dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
#     return dice
    
class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, pred, targ):
        pred = torch.sigmoid(pred)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
    
###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, bce_all, dice_all = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, masks) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks  = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        
        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
            
            bce_loss = losses_dict["BCELoss"](y_preds, masks)
            # dice_loss = 0.3 * losses_dict["DiceLoss"](y_preds, masks)
            losses = bce_loss # + dice_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item() 
        bce_all += bce_loss.item()
        # dice_all += dice_loss.item() 
    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.5f}".format(current_lr), flush=True)
    print("loss: {:.3f}, bce_all: {:.3f}, dice_all: {:.3f}".format(losses_all/len(train_loader), bce_all/len(train_loader), dice_all), flush=True)
        
@torch.no_grad()
def valid_one_epoch(model, valid_loader, metric, CFG):
    model.eval()
    metric.reset()
    
    val_images = []
    val_masks = []
    val_preds = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, masks) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks   = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        
        y_preds = model(images) 
        y_preds   = torch.nn.Sigmoid()(y_preds) # [b, c, w, h]
        
        val_masks.append(masks)
        val_preds.append(y_preds)
        val_images.append(images)
        metric.accumulate(y_preds.detach(), masks)
    
    val_images = torch.cat(val_images)
    val_masks = torch.cat(val_masks)
    val_preds = torch.cat(val_preds)
    val_images = val_images.cpu().numpy()
    val_masks = val_masks.cpu().numpy()
    val_preds = val_preds.cpu().numpy()
    
    val_dice = metric.value
    print("val_dice: {:.4f}".format(val_dice), flush=True)

    return val_dice, val_images, val_masks, val_preds


#iterator like wrapper that returns predicted masks
class test_one_epoch:
    def __init__(self, models, dl, tta:bool=True, half:bool=False, CFG=None):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half
        
    def __iter__(self):
        count=0
        with torch.no_grad():
            for x,y in iter(self.dl):
                if ((y>=0).sum() > 0): #exclude empty images
                    x = x[y>=0].to(CFG.device)
                    y = y[y>=0]
                    if self.half: x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None: py = p
                        else: py += p
                    if self.tta:
                        #x,y,xy flips as TTA
                        flips = [[-1],[-2],[-2,-1]]
                        for f in flips:
                            xf = torch.flip(x,f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p,f)
                                py += torch.sigmoid(p).detach()
                        py /= (1+len(flips))        
                    py /= len(self.models) # [bs, 1, 256, 256]
                    py = F.upsample(py, scale_factor=CFG.reduce, mode="bilinear")
                    py = py.permute(0,2,3,1).float().cpu()
                    
                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i],y[i]
                        count += 1
                    
    def __len__(self):
        return len(self.dl.dataset)
    

if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_worker = 16 # if debug
        data_path = "../input/hubmap-organ-segmentation"
        third_data_path = "../input/hubmap-2022-256x256"
        ckpt_path = "../input/ckpt-frank/resnet18_img512_bs8_fold4" # for submit
        # step2: data
        n_fold = 4
        img_size = [512, 512]
        train_bs = 8
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'resnet18'
        num_classes = 1
        # step4: optimizer
        epoch = 50
        lr = 1e-4
        wd = 1e-5
        lr_drop = 30
        # step5: infer
        thr = 0.3
        reduce = 4
        sz = 256
        s_th = 40  #saturation blancking threshold
        p_th = 1000*(sz//256)**2 #threshold for the minimum number of pixels
    
    set_seed(CFG.seed)
    if not os.path.exists(CFG.ckpt_path):
        os.makedirs(CFG.ckpt_path)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        df = pd.read_csv(os.path.join(CFG.data_path, "train.csv"))

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        df.loc[:,'fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df['id'], y=df['organ'])):
            df.iloc[val_idx, -1] = fold
        
        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataset_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader
            
            model = build_model(CFG) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd) # optimizer
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss
            metric=Dice_soft()
            
            best_val_dice = 0
            best_epoch = 0
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<d
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_dice, _, _, _ = valid_one_epoch(model, valid_loader, metric, CFG)
                
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_dice > best_val_dice)
                best_val_dice = max(best_val_dice, val_dice)
                if is_best:
                    save_path = f"{CFG.ckpt_path}/best_fold{fold}_epoch{epoch}_dice{best_val_dice:.4f}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_dice), flush=True)
    
    
    visual_flag = False
    if visual_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        df = pd.read_csv(os.path.join(CFG.data_path, "train.csv"))

        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        df.loc[:,'fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df['id'], y=df['organ'])):
            df.iloc[val_idx, -1] = fold
        
        ###############################################################
        ##### >>>>>>> step2: infer & visual <<<<<<
        ###############################################################
        # only save best checkpoint
        ##### cd ../input/ckpt-frank/resnet18_img512_bs8_fold2
        ##### cp best_fold0_epoch62_dice0.6791.pth best_flag_fold0_epoch62_dice0.6791.pth
        ##### cp best_fold1_epoch80_dice0.6604.pth best_flag_fold1_epoch80_dice0.6604.pth
        ##### rm -rf best_fold*
        for fold in range(CFG.n_fold):
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataset_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader & transform
            model = build_model(CFG, test_flag=True) # model
            
            sub_ckpt_path = "../input/ckpt-frank/resnet18_img512_bs8_fold2/best_flag_fold1_epoch80_dice0.6604.pth"
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            val_dice, val_images, val_masks, val_preds = valid_one_epoch(model, valid_loader, CFG)
            
            for idx in range(val_images.shape[0]):
                plot_visual(val_images[idx], val_masks[idx], val_preds[idx], fold, idx, "bwr")
                
        pdb.set_trace()
        
        
    test_flag = False
    if test_flag:
        set_seed(CFG.seed)
        
        ###############################################################
        ##### part0: load model
        ###############################################################
        # attention: change the corresponding upload path to kaggle!!!!!
        ckpt_paths  = glob.glob(f'{CFG.ckpt_path}/best_flag_*')
        assert len(ckpt_paths) == CFG.n_fold, "ckpt path error!"
        
        models = []
        for sub_ckpt_path in ckpt_paths:
            state_dict = torch.load(sub_ckpt_path, map_location=torch.device('cpu'))
            model = smp.Unet(
                encoder_name='resnet18',
                encoder_weights=None, 
                in_channels=3,             
                classes=1,   
                activation=None,
            )
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
            model.to(CFG.device)
            models.append(model)
        
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        df_sample = pd.read_csv(os.path.join(CFG.data_path, "sample_submission.csv"))
        
        
        sz = 256    # the size of tiles
        reduce = 4  # reduce the original images by 4 times
        TH = 0.225  # threshold for positive predictions
        s_th = 40  #saturation blancking threshold
        p_th = 1000*(sz//256)**2 #threshold for the minimum number of pixels
        identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

        names,preds = [],[]
        for idx,row in tqdm(df_sample.iterrows(),total=len(df_sample)):
            idx = str(row['id'])
            ds = build_dataset(idx=idx, label=False, sz=sz, reduce=reduce, CFG=CFG)
            #rasterio cannot be used with multiple workers
            dl = DataLoader(ds, CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
            mp = test_one_epoch(models,dl)
            #generate masks
            mask = torch.zeros(len(ds),ds.sz,ds.sz,dtype=torch.int8)
            for p,i in iter(mp): mask[i.item()] = p.squeeze(-1) > TH
            
            #reshape tiled masks into a single mask and crop padding
            mask = mask.view(ds.n0max,ds.n1max,ds.sz,ds.sz).\
                permute(0,2,1,3).reshape(ds.n0max*ds.sz,ds.n1max*ds.sz)
            mask = mask[ds.pad0//2:-(ds.pad0-ds.pad0//2) if ds.pad0 > 0 else ds.n0max*ds.sz,
                ds.pad1//2:-(ds.pad1-ds.pad1//2) if ds.pad1 > 0 else ds.n1max*ds.sz]
            
            #convert to rle
            #https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
            rle = rle_encode_less_memory(mask.numpy())
            names.append(idx)
            preds.append(rle)
      

        df = pd.DataFrame({'id':names,'rle':preds})
        df.to_csv('submission.csv',index=False)


        